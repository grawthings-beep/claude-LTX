# claude-LTX

RunPod ComfyUI template for LTX 2.3 I2V, rebuilt around the 10Eros checkpoint
and stronger image-reference workflows.

This repo intentionally does not default to the plain official LTX 2.3 example
workflows. The recommended workflows use:

- `10Eros_v1-fp8mixed_learned.safetensors` from `TenStrip/LTX2.3-10Eros`
- TenStrip distilled LoRA `condsafe`
- `LTX2.3_reasoning_I2V_V3.safetensors`
- `ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors` in the
  `00_` and `01_` recommended workflows
- source-image guide reapplication in the second pass
- DaSiWa-style fast scheduler settings for clearer I2V output

## RunPod Image

```text
ghcr.io/grawthings-beep/claude-ltx:cuda12.8
```

Expose HTTP port `8188`, mount the persistent volume at `/workspace`, and use
the env vars from `runpod-template.env.example`.

## Recommended Workflows

Use these first:

```text
00_recommended_i2v_identity_lock_10eros.json
01_recommended_i2v_simple_10eros.json
```

`00_recommended_i2v_identity_lock_10eros.json` uses the same input image as both
the first-frame and last-frame guide. It is the best default when the source
character keeps drifting into a different person.

`01_recommended_i2v_simple_10eros.json` is the lighter single-image I2V path.
Use it when motion is more important than loop/identity locking.

The older Civitai-assisted workflows are still included, but they need
`CIVITAI_TOKEN` and are not the first recommendation.

## Model Storage

Models are downloaded into `/workspace/comfyui/models`, so a RunPod persistent
volume or Network Volume will reuse them across Pod restarts.

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
