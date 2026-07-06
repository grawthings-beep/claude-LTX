#!/usr/bin/env python3
"""Generate a long video by chaining LTX 2.3 I2V segments through the ComfyUI API.

Each segment is generated with an API-format workflow (exported once from the
ComfyUI UI). The last frame of segment N becomes the input image of segment
N+1. Finally all segments are concatenated with ffmpeg, dropping the
duplicated boundary frame of every segment after the first.

This is pixel-space chaining, not latent-space video extension. Known
limitations are documented in the repository README.

Only the Python standard library is required. ffmpeg/ffprobe must be in PATH.
"""
import argparse
import copy
import json
import mimetypes
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
DEFAULT_PROMPT_TITLE = "CLIP Text Encode (Positive Prompt)"
DEFAULT_FRAMES_TITLE = "number of frames"


def log(message):
    print(message, flush=True)


def die(message):
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Workflow patching (pure functions, unit-tested)
# ---------------------------------------------------------------------------

def nodes_by_class(workflow, class_type):
    return sorted(
        (nid for nid, node in workflow.items() if node.get("class_type") == class_type),
        key=lambda nid: int(nid) if str(nid).isdigit() else 0,
    )


def nodes_by_title(workflow, title):
    return sorted(
        (
            nid
            for nid, node in workflow.items()
            if node.get("_meta", {}).get("title") == title
        ),
        key=lambda nid: int(nid) if str(nid).isdigit() else 0,
    )


def validate_frames(frames):
    if frames < 9 or (frames - 1) % 8 != 0:
        raise ValueError(
            f"frames must be 8*k+1 (e.g. 121, 241, 361), got {frames}"
        )
    return frames


def patch_workflow(workflow, *, image=None, prompt=None, frames=None, seed=None,
                   prompt_title=DEFAULT_PROMPT_TITLE, frames_title=DEFAULT_FRAMES_TITLE):
    """Return a patched copy of an API-format workflow dict."""
    wf = copy.deepcopy(workflow)

    if image is not None:
        load_nodes = nodes_by_class(wf, "LoadImage")
        if not load_nodes:
            raise ValueError("workflow has no LoadImage node; export an I2V workflow")
        if len(load_nodes) > 1:
            log(f"WARN: {len(load_nodes)} LoadImage nodes found, patching all of them")
        for nid in load_nodes:
            wf[nid]["inputs"]["image"] = image
        # The official LTX 2.3 T2V/I2V workflow has a "bypass_i2v" toggle
        # that, when True, silently ignores the loaded image (T2V mode).
        # An input image only makes sense in I2V mode, so force it off.
        for nid in nodes_by_title(wf, "bypass_i2v"):
            if "value" in wf[nid]["inputs"]:
                wf[nid]["inputs"]["value"] = False

    if prompt is not None:
        targets = nodes_by_title(wf, prompt_title)
        if not targets:
            raise ValueError(
                f"no node titled {prompt_title!r}; set --prompt-title to the "
                "title of your positive prompt node"
            )
        for nid in targets:
            inputs = wf[nid]["inputs"]
            for key in ("text", "prompt", "value", "string"):
                if key in inputs and isinstance(inputs[key], str):
                    inputs[key] = prompt
                    break
            else:
                raise ValueError(f"node {nid} ({prompt_title!r}) has no text input")

    if frames is not None:
        validate_frames(frames)
        targets = nodes_by_title(wf, frames_title)
        if targets:
            for nid in targets:
                wf[nid]["inputs"]["value"] = frames
        else:
            fallback = nodes_by_class(wf, "EmptyLTXVLatentVideo")
            patched = False
            for nid in fallback:
                if isinstance(wf[nid]["inputs"].get("length"), int):
                    wf[nid]["inputs"]["length"] = frames
                    patched = True
            if not patched:
                raise ValueError(
                    f"no node titled {frames_title!r} and no patchable "
                    "EmptyLTXVLatentVideo.length; cannot set frame count"
                )

    if seed is not None:
        for offset, nid in enumerate(nodes_by_class(wf, "RandomNoise")):
            wf[nid]["inputs"]["noise_seed"] = seed + offset
        for offset, nid in enumerate(nodes_by_class(wf, "KSampler")):
            wf[nid]["inputs"]["seed"] = seed + offset

    return wf


def find_video_outputs(history_entry):
    """Extract (filename, subfolder, type) tuples for video files from a
    ComfyUI /history entry."""
    results = []
    for output in history_entry.get("outputs", {}).values():
        for value in output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename", "")
                ext = pathlib.Path(filename).suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    results.append(
                        (filename, item.get("subfolder", ""), item.get("type", "output"))
                    )
    return results


# ---------------------------------------------------------------------------
# ComfyUI API client (stdlib only)
# ---------------------------------------------------------------------------

def http_json(url, payload=None, timeout=60):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def upload_image(server, path):
    path = pathlib.Path(path)
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            path.read_bytes(),
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue',
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{server}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        info = json.loads(response.read().decode("utf-8"))
    name = info.get("name", path.name)
    subfolder = info.get("subfolder", "")
    return f"{subfolder}/{name}" if subfolder else name


def queue_prompt(server, workflow, client_id):
    result = http_json(
        f"{server}/prompt", {"prompt": workflow, "client_id": client_id}
    )
    if "prompt_id" not in result:
        raise RuntimeError(f"unexpected /prompt response: {result}")
    return result["prompt_id"]


def wait_for_prompt(server, prompt_id, timeout, poll_interval=5):
    started = time.monotonic()
    while True:
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout}s")
        try:
            history = http_json(f"{server}/history/{prompt_id}", timeout=30)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(poll_interval)
            continue
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise RuntimeError(f"prompt {prompt_id} failed: {messages}")
            if entry.get("outputs"):
                return entry
        time.sleep(poll_interval)


def download_output(server, filename, subfolder, folder_type, destination):
    query = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{server}/view?{query}", timeout=600) as response, \
            destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    return destination


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def run(cmd):
    log("RUN: " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def probe(path, entries, select="v:0"):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", select,
            "-show_entries", entries,
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return out


def video_fps(path):
    rate = probe(path, "stream=r_frame_rate")
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1)


def video_frame_count(path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


def has_audio(path):
    return bool(probe(path, "stream=codec_type", select="a"))


def extract_last_frame(video, destination):
    count = video_frame_count(video)
    run([
        "ffmpeg", "-y", "-v", "error", "-i", video,
        "-vf", f"select=eq(n\\,{count - 1})", "-vframes", "1",
        "-update", "1", destination,
    ])
    if not pathlib.Path(destination).exists():
        raise RuntimeError(f"failed to extract last frame from {video}")
    return destination


def concat_segments(segments, output, crf=16):
    """Re-encode segments uniformly (dropping frame 0 of segments 2..N,
    which duplicates the previous segment's last frame) and concat."""
    workdir = pathlib.Path(output).parent / (pathlib.Path(output).stem + "_parts")
    workdir.mkdir(parents=True, exist_ok=True)
    fps = video_fps(segments[0])
    audio = has_audio(segments[0])
    parts = []
    for index, segment in enumerate(segments):
        part = workdir / f"part_{index:03d}.mp4"
        skip = 0 if index == 0 else 1
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", segment]
        vf = f"select=gte(n\\,{skip}),setpts=N/{fps}/TB"
        cmd += ["-vf", vf]
        if audio:
            cmd += ["-af", f"atrim=start={skip / fps},asetpts=PTS-STARTPTS"]
        cmd += ["-r", f"{fps}", "-c:v", "libx264", "-crf", str(crf),
                "-preset", "medium", "-pix_fmt", "yuv420p"]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += [part]
        run(cmd)
        parts.append(part)

    list_file = workdir / "concat.txt"
    list_file.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8"
    )
    run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output,
    ])
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188",
                        help="ComfyUI base URL")
    parser.add_argument("--workflow", required=True,
                        help="API-format workflow JSON (ComfyUI: Export (API))")
    parser.add_argument("--image", required=True,
                        help="input image for the first segment")
    parser.add_argument("--prompt", action="append", default=[],
                        help="positive prompt; repeat once per segment "
                             "(last one is reused if fewer than --segments)")
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--frames", type=int, default=None,
                        help="frames per segment, must be 8*k+1 (121, 241, ...)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=1000,
                        help="seed increment between segments")
    parser.add_argument("--prompt-title", default=DEFAULT_PROMPT_TITLE)
    parser.add_argument("--frames-title", default=DEFAULT_FRAMES_TITLE)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="max seconds to wait per segment")
    parser.add_argument("--output", required=True, help="final mp4 path")
    parser.add_argument("--workdir", default=None,
                        help="directory for segment files "
                             "(default: <output>_segments)")
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--keep-segments", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.segments < 1:
        die("--segments must be >= 1")
    if not args.prompt:
        die("at least one --prompt is required")
    if args.frames is not None:
        try:
            validate_frames(args.frames)
        except ValueError as exc:
            die(str(exc))
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(f"{tool} not found in PATH")

    workflow = json.loads(pathlib.Path(args.workflow).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or "nodes" in workflow:
        die("workflow must be API format (use Export (API) in ComfyUI, "
            "not the regular Save)")

    output = pathlib.Path(args.output)
    workdir = pathlib.Path(args.workdir) if args.workdir else \
        output.parent / (output.stem + "_segments")
    workdir.mkdir(parents=True, exist_ok=True)
    client_id = uuid.uuid4().hex

    current_image = pathlib.Path(args.image)
    segment_files = []
    for index in range(args.segments):
        prompt = args.prompt[min(index, len(args.prompt) - 1)]
        seed = args.seed + index * args.seed_step
        log(f"--- segment {index + 1}/{args.segments} "
            f"(seed {seed}, image {current_image.name}) ---")

        uploaded = upload_image(args.server, current_image)
        patched = patch_workflow(
            workflow, image=uploaded, prompt=prompt, frames=args.frames,
            seed=seed, prompt_title=args.prompt_title,
            frames_title=args.frames_title,
        )
        prompt_id = queue_prompt(args.server, patched, client_id)
        log(f"queued prompt {prompt_id}")
        entry = wait_for_prompt(args.server, prompt_id, args.timeout)

        videos = find_video_outputs(entry)
        if not videos:
            die(f"segment {index + 1}: no video output found in history")
        filename, subfolder, folder_type = videos[0]
        segment_path = workdir / f"segment_{index:03d}{pathlib.Path(filename).suffix}"
        download_output(args.server, filename, subfolder, folder_type, segment_path)
        log(f"segment saved: {segment_path}")
        segment_files.append(segment_path)

        if index + 1 < args.segments:
            next_image = workdir / f"segment_{index:03d}_last.png"
            extract_last_frame(segment_path, next_image)
            current_image = next_image

    log("--- concatenating segments ---")
    concat_segments(segment_files, output, crf=args.crf)
    log(f"DONE: {output}")

    if not args.keep_segments:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
