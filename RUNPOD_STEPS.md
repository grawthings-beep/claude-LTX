# RunPod setup

## 1. Wait for the GHCR build

After pushing to `main`, wait until this Actions workflow is green:

```text
https://github.com/grawthings-beep/claude-LTX/actions
```

Use this image:

```text
ghcr.io/grawthings-beep/claude-ltx:cuda12.8
```

## 2. Template settings

```text
Container Image: ghcr.io/grawthings-beep/claude-ltx:cuda12.8
HTTP Port: 8188
Container Disk: 40 GB+
Volume / Network Volume: 180 GB+
Volume Mount Path: /workspace
```

Leave the command/start command empty. The image starts
`/opt/claude-ltx/scripts/start.sh`.

## 3. Environment variables

```env
PORT=8188
LISTEN=0.0.0.0
RUN_DEP_CHECK=0
DOWNLOAD_MODELS=1
MODEL_DOWNLOAD_MODE=background
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
MODEL_MANIFEST_URL=https://raw.githubusercontent.com/grawthings-beep/claude-LTX/main/config/ltx-video-models.json
HF_XET_HIGH_PERFORMANCE=1
HF_HUB_DOWNLOAD_TIMEOUT=120
ARIA2_CONNECTIONS=8
ARIA2_SPLITS=8
DOWNLOAD_JOBS=1
VERIFY_MODEL_HASHES=once
COMFYUI_ARGS=--reserve-vram 5
```

No `CIVITAI_TOKEN` is required for the bundled workflows. Civitai LoRAs remain
in the manifest as optional downloads and are skipped unless `CIVITAI_TOKEN` is
set.

## 4. First boot

ComfyUI opens while models download in the background. Watch:

```bash
cat /workspace/comfyui/logs/model-download.status
tail -f /workspace/comfyui/logs/model-download.log
```

When the status is `complete`, refresh ComfyUI.

Use blocking mode if you want the port to open only after downloads complete:

```env
MODEL_DOWNLOAD_MODE=blocking
```

## 5. Workflow choice

Use the normal `1728x1152` I2V workflow:

```text
i2v.json
```

Replace the `LoadImage` image with your own source image.

Use the experimental natural-loop workflow separately:

```text
loop.json
```

The loop workflow uses the bundled LTX Mobius sampler, cyclic VAE decode, and
audio seam correction. It is independent from the normal I2V workflow.
