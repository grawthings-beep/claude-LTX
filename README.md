# claude-LTX

RunPod ComfyUI template for LTX I2V workflows.

The bundled workflow set is intentionally small:

- `original.json`
- `i2v.json`
- `loop.json`

`original.json` preserves the uploaded reference graph, including its optional
AnimeSharp and RIFE branch. `i2v.json` keeps that graph but disables the slow
RIFE side output. Both use the normal `1728x1152` I2V path, the 10Eros checkpoint,
the official distilled 384 LoRA at `0.5`, and the official LTX spatial x2
upscaler. Its audio VAE, vocoder, and text projection are loaded from the same
10Eros checkpoint, avoiding a redundant 27 GiB dev checkpoint download.

`loop.json` is the natural-loop path. Its first pass keeps the normal two-stage
I2V graph and generates 129 frames without forcing the input image onto the
last frame. A second two-stage LTX pass generates a 49-frame motion bridge,
conditioned on the first pass's final 9 frames and initial 9 frames. Both bridge
stages crop their guide latents before decode. The assembler removes repeated
guide contexts and returns 160 frames at 24 fps, while smoothing the generated
audio at the internal join and loop boundary.

## RunPod Image

```text
ghcr.io/grawthings-beep/claude-ltx:cuda13.0
```

Expose HTTP port `8188`, mount the persistent volume at `/workspace`, and use
the env vars from `runpod-template.env.example`.

The image uses a pinned CUDA 13.0 RunPod base with PyTorch cu130 for RTX 5090
support and bundles only the external node pack used by the workflows. At
startup it repairs missing NVIDIA UVM device nodes when the container permits
it, then performs a low-level CUDA driver probe. ComfyUI starts immediately
while model downloads continue in the background. The `cuda12.8` tag remains
as a compatibility alias for existing templates.

## Model Storage

Models are downloaded into `/workspace/comfyui/models`, so a RunPod persistent
volume or Network Volume will reuse them across Pod restarts. A Network Volume
mounted at `/workspace` is required to preserve this cache when replacing a Pod.

Models referenced by the bundled workflows have download priority. Other LoRAs
remain in the manifest and download afterward. Fresh Hugging Face Xet downloads
whose content hash matches the manifest are not read in full a second time.

HF-hosted LoRAs used by the bundled workflows are downloaded automatically when
`HF_TOKEN` is set. Existing Civitai LoRAs are still kept as optional entries and
are skipped unless `CIVITAI_TOKEN` is provided.

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
