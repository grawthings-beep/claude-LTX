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

`loop.json` is the natural-loop path. It keeps the original two-stage graph and
uses ComfyUI's built-in `LTXVAddGuide` at frames `0` and `-1` in both stages,
with the same input image at strength `0.7`. This follows the official LTX
FLF2V conditioning pattern instead of overwriting the final frame. A small
bundled node smooths the generated audio boundary without changing its duration.

## RunPod Image

```text
ghcr.io/grawthings-beep/claude-ltx:cuda12.8
```

Expose HTTP port `8188`, mount the persistent volume at `/workspace`, and use
the env vars from `runpod-template.env.example`.

The image uses a pinned CUDA 12.8 RunPod base for broad host-driver compatibility
and bundles only the external node pack used by the workflows. ComfyUI checks
the NVIDIA driver directly before starting, without repeatedly importing
PyTorch. Set `WAIT_FOR_GPU=0` only for intentional CPU-only diagnostics.

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
