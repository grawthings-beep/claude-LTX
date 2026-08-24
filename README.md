# claude-LTX

RunPod ComfyUI template for the MrXin LTX 2.3 I2V EROS workflow.

Only one workflow is bundled:

- `mrxin-i2v.json`

`mrxin-i2v.json` keeps the graph from Civitai model version `2835183`
(`mrxinLTX23I2VEros12GBVRAM_i2vV40.zip`). It includes the checkpoint/distilled
model switch, two-stage I2V path, audio path, optional editor, RIFE interpolation,
and optional RTX video super resolution.

The RunPod defaults use the complete 10Eros checkpoint branch, TenStrip's
I2V-safe `condsafe` distilled LoRA, five-second 960x1280 generation, official
first-pass I2V conditioning strength, and the VAE last-frame artifact fix.
Content LoRAs remain available but start disabled so combinations can be tested
one at a time.
The standalone audio VAE is stored under `models/checkpoints`, which is the
directory read by ComfyUI's `LTXVAudioVAELoader`.

## RunPod Image

```text
ghcr.io/grawthings-beep/claude-ltx:cuda13.0
```

Each build also publishes an immutable image tag using the full Git commit SHA.
Use that tag for RunPod deployments when a worker may have cached a mutable tag.
The startup log prints `claude-LTX image revision` so the running image can be
matched to the requested tag.

Expose HTTP port `8188`, mount the persistent volume at `/workspace`, and use
the env vars from `runpod-template.env.example`.

The image uses a pinned CUDA 13.0 RunPod base with PyTorch cu130 for RTX 5090
support and bundles the external node packs required by the workflow. At
startup it repairs missing NVIDIA UVM device nodes when the container permits
it, removes the retired bundled `i2v.json`, `original.json`, and `loop.json`
files, installs `mrxin-i2v.json`, then performs a low-level CUDA driver probe.
ComfyUI starts immediately
while model downloads continue in the background. The `cuda12.8` tag remains
as a compatibility alias for existing templates.

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
