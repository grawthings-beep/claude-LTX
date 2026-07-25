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

Use the reference workflow for the `1152x896` setup from the sample MP4:

```text
02_reference_ltx23_i2v_1152x896_phut_hon.json
```

Replace the `LoadImage` image with your own source image.

The reference workflow preserves the RIFE frame interpolation branch from the
source JSON, so the image installs `ComfyUI-Frame-Interpolation` and `rife49.pth`
during the image build. The branch's AnimeSharpV4 x2 RCAN upscaler is downloaded
through the model manifest. The RIFE/VHS branch is disabled by default; use the
main `SaveVideo` output for normal-speed video with audio.

Use the simple 10Eros fallback when you want fewer LoRA dependencies:

```text
01_recommended_i2v_simple_10eros.json
```
