# claude-LTX

RunPod ComfyUI template for the MrXin LTX 2.3 I2V EROS workflow.

Four workflows are bundled:

- `mrxin-i2v.json`
- `mrxin-i2v-hq.json`
- `mrxin-i2v-auto-mosaic.json`
- `mrxin-i2v-2stage-auto-mosaic.json`

`mrxin-i2v.json` keeps the graph from Civitai model version `2835183`
(`mrxinLTX23I2VEros12GBVRAM_i2vV40.zip`). It includes the checkpoint/distilled
model switch, two-stage I2V path, audio path, optional editor, RIFE interpolation,
and optional RTX video super resolution.

The RunPod defaults use the complete 10Eros checkpoint branch, TenStrip's
I2V-safe `condsafe` distilled LoRA, five-second 960x1280 generation, official
first-pass I2V conditioning strength, and the VAE last-frame artifact fix.
Content LoRAs remain available but start disabled so combinations can be tested
one at a time.
`mrxin-i2v-hq.json` is an experimental high-resolution copy. It generates the
first pass at 896x1184 and uses the latent x2 stage for a 1792x2368 final video.
Its I2V image path bypasses the original 1536-pixel longer-edge reduction so
the final pass receives the full-resolution conditioning image.
`mrxin-i2v-auto-mosaic.json` keeps only that workflow's 896x1184 first pass;
it does not run the latent x2 stage. After VAE decode, a CPU-only YOLO11
instance-segmentation node applies the JUST contour mosaic once, immediately
before MP4 encoding. The input image is never mosaicked. Default targets are
`pussy`, `penis`, and `testicles`; `anus` is excluded.
`mrxin-i2v-2stage-auto-mosaic.json` preserves both HQ generation stages and
the latent x2 step. It applies the same CPU JUST contour mosaic once after the
second-pass VAE decode and immediately before the final MP4 encoder. The
first-pass preview path and input image are unchanged.
The standalone audio VAE is stored under `models/checkpoints`, which is the
directory read by ComfyUI's `LTXVAudioVAELoader`.

## RunPod Image

```text
ghcr.io/grawthings-beep/claude-ltx:cuda12.8
```

CUDA 12.8 is the default and supports RTX 5090/B200 hosts running the common
R570 driver. `latest`, `cuda12.8`, the full Git commit SHA, and
`<SHA>-cuda12.8` all refer to the actual CUDA 12.8 build. CUDA 13.0 is published
separately as `cuda13.0` and `<SHA>-cuda13.0`; it requires an R580 or newer host
driver. Never use the CUDA 13.0 image on a host that reports CUDA 12.8.
Use an immutable SHA tag for RunPod deployments when a worker may have cached a
mutable tag.
The startup log prints `claude-LTX image revision` so the running image can be
matched to the requested tag.

Expose HTTP port `8188`, mount the persistent volume at `/workspace`, and use
the env vars from `runpod-template.env.example`.

Both images use pinned RunPod bases and bundle the external node packs required
by the workflow. At startup the container repairs missing NVIDIA UVM device
nodes when permitted, then verifies both the low-level CUDA driver and the
actual PyTorch CUDA runtime before downloading models. An incompatible host
therefore stops immediately with a clear CUDA 12.8/R570 or CUDA 13.0/R580
message.
The required 18 MB auto-mosaic segmentation archive is downloaded, size- and
SHA256-verified, and extracted before ComfyUI starts. All larger model downloads
continue in the background.

## Model Storage

Models are downloaded into `/workspace/comfyui/models`, so a RunPod persistent
volume or Network Volume will reuse them across Pod restarts. A Network Volume
mounted at `/workspace` is required to preserve this cache when replacing a Pod.

Models referenced by the bundled workflow have download priority. Other LoRAs
remain in the manifest and download afterward. Fresh Hugging Face Xet downloads
whose content hash matches the manifest are not read in full a second time.

The requested Civitai LoRA (`fileId=2736052`) is the first manifest entry and is
downloaded when `CIVITAI_TOKEN` is set. Its Hugging Face backup and all existing
LoRA entries remain in the manifest.

Set `CIVITAI_API_TOKEN` as a RunPod Secret. It authenticates only the required
Anime NSFW Detection v5.0 archive request and is never stored in the repository,
logs, or image. A Pod without the extracted model and this Secret stops before
ComfyUI starts, instead of loading a broken auto-mosaic workflow.

Useful logs:

```bash
cat /workspace/comfyui/logs/model-download.status
tail -f /workspace/comfyui/logs/model-download.log
```

## Long Video Script

Export a workflow from ComfyUI with `Export (API)`, then run:

```bash
python3 /opt/claude-ltx/scripts/long_video.py \
  --server http://127.0.0.1:8188 \
  --workflow /workspace/comfyui/exports/two_stage_api.json \
  --image /workspace/comfyui/input/start.png \
  --segments 3 --frames 241 --seed 42 \
  --prompt "segment 1 prompt..." \
  --prompt "segment 2 prompt..." \
  --prompt "segment 3 prompt..." \
  --output /workspace/comfyui/output/long_video.mp4
```

Start with `--segments 2` before scaling up. Segment stitching is pixel-space
I2V chaining, not true latent video extension, so prompt continuity still matters.
