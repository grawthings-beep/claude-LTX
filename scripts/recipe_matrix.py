#!/usr/bin/env python3
"""Sweep LTX 2.3 recipe parameters and generate comparison clips.

Runs the same API-format workflow repeatedly over a cartesian product of
parameter values (any node input, addressed by node title), with a fixed
seed and input image, so differences between outputs come only from the
swept parameters. Produces one video per combination plus an index.html
for side-by-side comparison.

Example:

    python3 recipe_matrix.py \
      --workflow two_stage_api.json --image start.png \
      --prompt "..." --seed 42 --frames 121 \
      --axis "denoise=BasicScheduler:denoise=0.3,0.42,0.55" \
      --axis "lora=LoraLoaderModelOnly:strength_model=0.5,0.75,1.0" \
      --set "SaveVideo:codec=h264" \
      --output-dir /workspace/comfyui/output/matrix01

Only the Python standard library is required.
"""
import argparse
import itertools
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import long_video  # noqa: E402


def parse_value(text):
    lowered = text.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def parse_set(spec):
    """'NodeTitle:input=value' -> (title, input_name, value)"""
    head, sep, raw_value = spec.partition("=")
    if not sep:
        raise ValueError(f"invalid --set {spec!r}, expected 'Title:input=value'")
    title, sep, input_name = head.rpartition(":")
    if not sep or not title or not input_name:
        raise ValueError(f"invalid --set {spec!r}, expected 'Title:input=value'")
    return title, input_name, parse_value(raw_value)


def parse_axis(spec):
    """'name=NodeTitle:input=v1,v2' -> (name, title, input_name, [values])"""
    name, sep, rest = spec.partition("=")
    if not sep or not name or "=" not in rest:
        raise ValueError(
            f"invalid --axis {spec!r}, expected 'name=Title:input=v1,v2,...'"
        )
    title, input_name, raw = parse_set_axis(rest)
    values = [parse_value(v) for v in raw.split(",") if v.strip() != ""]
    if not values:
        raise ValueError(f"axis {name!r} has no values")
    return name.strip(), title, input_name, values


def parse_set_axis(spec):
    head, sep, raw_value = spec.partition("=")
    if not sep:
        raise ValueError(f"invalid axis target {spec!r}")
    title, sep, input_name = head.rpartition(":")
    if not sep or not title or not input_name:
        raise ValueError(f"invalid axis target {spec!r}")
    return title, input_name, raw_value


def apply_overrides(workflow, overrides):
    """Set node inputs by (title, input_name, value). Raises if the title or
    input does not exist, so typos fail loudly instead of silently doing
    nothing."""
    for title, input_name, value in overrides:
        nids = long_video.nodes_by_title(workflow, title)
        if not nids:
            raise ValueError(f"no node titled {title!r} in workflow")
        for nid in nids:
            inputs = workflow[nid]["inputs"]
            if input_name not in inputs:
                raise ValueError(
                    f"node {title!r} has no input {input_name!r}; "
                    f"available: {sorted(inputs)}"
                )
            inputs[input_name] = value
    return workflow


def combo_label(names, values):
    parts = []
    for name, value in zip(names, values):
        token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
        parts.append(f"{name}-{token}")
    return "__".join(parts) if parts else "single"


def write_index_html(output_dir, results):
    cells = []
    for label, filename in results:
        cells.append(
            f'<figure><video src="{filename}" controls loop muted '
            f'preload="metadata"></video><figcaption>{label}</figcaption></figure>'
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>recipe matrix</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}"
        "video{width:100%}figure{margin:0}figcaption{font-size:12px;word-break:break-all}"
        "</style><main>" + "".join(cells) + "</main>"
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--workflow", required=True,
                        help="API-format workflow JSON (ComfyUI: Export (API))")
    parser.add_argument("--image", default=None,
                        help="input image (I2V); omit for T2V workflows")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="fixed seed shared by every combination")
    parser.add_argument("--frames", type=int, default=None,
                        help="frames per clip, 8*k+1; keep short (e.g. 121)")
    parser.add_argument("--axis", action="append", default=[],
                        help="'name=NodeTitle:input=v1,v2,...' (repeatable; "
                             "cartesian product across axes)")
    parser.add_argument("--set", dest="sets", action="append", default=[],
                        help="'NodeTitle:input=value' applied to every run")
    parser.add_argument("--prompt-title", default=long_video.DEFAULT_PROMPT_TITLE)
    parser.add_argument("--frames-title", default=long_video.DEFAULT_FRAMES_TITLE)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="print combinations and patched values, "
                             "do not contact the server")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    axes = [parse_axis(spec) for spec in args.axis]
    fixed = [parse_set(spec) for spec in args.sets]

    workflow = json.loads(pathlib.Path(args.workflow).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or "nodes" in workflow:
        long_video.die("workflow must be API format (Export (API) in ComfyUI)")

    names = [axis[0] for axis in axes]
    value_lists = [axis[3] for axis in axes]
    combos = list(itertools.product(*value_lists)) if axes else [()]
    log = long_video.log
    log(f"{len(combos)} combination(s)")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uploaded = None
    if args.image and not args.dry_run:
        uploaded = long_video.upload_image(args.server, args.image)

    import uuid
    client_id = uuid.uuid4().hex
    results = []
    for values in combos:
        label = combo_label(names, values)
        overrides = fixed + [
            (axes[i][1], axes[i][2], values[i]) for i in range(len(axes))
        ]
        patched = long_video.patch_workflow(
            workflow,
            image=uploaded if args.image else None,
            prompt=args.prompt,
            frames=args.frames,
            seed=args.seed,
            prompt_title=args.prompt_title,
            frames_title=args.frames_title,
        )
        apply_overrides(patched, overrides)

        if args.dry_run:
            log(f"DRY: {label}: " + ", ".join(
                f"{t}:{i}={v}" for t, i, v in overrides))
            continue

        log(f"--- {label} ---")
        prompt_id = long_video.queue_prompt(args.server, patched, client_id)
        entry = long_video.wait_for_prompt(args.server, prompt_id, args.timeout)
        videos = long_video.find_video_outputs(entry)
        if not videos:
            long_video.die(f"{label}: no video output in history")
        filename, subfolder, folder_type = videos[0]
        target = output_dir / f"{label}{pathlib.Path(filename).suffix}"
        long_video.download_output(
            args.server, filename, subfolder, folder_type, target
        )
        log(f"saved {target}")
        results.append((label, target.name))

    if not args.dry_run:
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "frames": args.frames,
                    "prompt": args.prompt,
                    "axes": [
                        {"name": a[0], "node": a[1], "input": a[2], "values": a[3]}
                        for a in axes
                    ],
                    "results": [
                        {"label": label, "file": name} for label, name in results
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_index_html(output_dir, results)
        log(f"DONE: {output_dir}/index.html")


if __name__ == "__main__":
    main()
