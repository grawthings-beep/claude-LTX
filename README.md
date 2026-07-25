# claude-LTX

RunPod ComfyUI template for LTX I2V workflows.

The bundled workflow set is intentionally small:

- `i2v.json`
- `loop.json`

`i2v.json` is the normal `1728x1152` I2V path. It uses the 10Eros checkpoint,
the official distilled 384 LoRA at `0.5`, the LTX 2.3 dev fp8 audio/text stack,
and the official LTX spatial x2 upscaler.

`loop.json` is the experimental natural-loop path. Its bundled custom nodes port
Mobius-style cyclic latent shifting to LTX, rotate video and audio together,
decode with wrapped temporal context, and smooth the final audio boundary.

## RunPod Image

```text
ghcr.io/grawthings-beep/claude-ltx:cuda12.8
```

Expose HTTP port `8188`, mount the persistent volume at `/workspace`, and use
the env vars from `runpod-template.env.example`.

The image uses a pinned, smaller CUDA 12.8 RunPod base and bundles only the
external node pack used by the workflows. ComfyUI waits for both `nvidia-smi`
and PyTorch CUDA initialization before starting, avoiding crash loops while a
RunPod GPU is still being attached. Set `WAIT_FOR_GPU=0` only for intentional
CPU-only diagnostics.

## Model Storage

Models are downloaded into `/workspace/comfyui/models`, so a RunPod persistent
volume or Network Volume will reuse them across Pod restarts.

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
