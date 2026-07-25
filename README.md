# claude-LTX

RunPod ComfyUI template for LTX I2V workflows.

The bundled workflow set is intentionally small:

- `01_recommended_i2v_simple_10eros.json`
- `02_reference_ltx23_i2v_1152x896_phut_hon.json`

`01_recommended_i2v_simple_10eros.json` is the 10Eros-based simple I2V path.
Use it as the stable fallback.

`02_reference_ltx23_i2v_1152x896_phut_hon.json` is imported from the reference
MP4 workflow. It uses the official LTX 2.3 fp8 checkpoint, Phut hon LoRA, and
Image2Vid Adapter LoRA, with `1152x896` output settings. The imported workflow
also keeps its RIFE frame interpolation branch, so the image includes the
`ComfyUI-Frame-Interpolation` node pack, `rife49.pth`, and the AnimeSharpV4 x2
RCAN upscaler used by that branch.

## RunPod Image

```text
ghcr.io/grawthings-beep/claude-ltx:cuda12.8
```

Expose HTTP port `8188`, mount the persistent volume at `/workspace`, and use
the env vars from `runpod-template.env.example`.

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
