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

If RunPod cannot pull it, make the GHCR package public or add GHCR registry
credentials in RunPod.

## 2. Template settings

```text
Container Image: ghcr.io/grawthings-beep/claude-ltx:cuda12.8
HTTP Port: 8188
Container Disk: 40 GB+
Volume / Network Volume: 150 GB+
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

Optional, only if you want the Civitai workflows:

```env
CIVITAI_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}
```

The recommended workflows do not require Civitai. They do require Hugging Face
access because some LoRAs are in the private `uwgm/nikke-loras` repo.

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

Start with:

```text
00_recommended_i2v_identity_lock_10eros.json
```

Replace the `LoadImage` image with your own source image. This workflow uses the
same source image as first and last guide, which is better when the character is
turning into a different person.

If it is too constrained, try:

```text
01_recommended_i2v_simple_10eros.json
```

For two separate keyframes, use:

```text
video_ltx23_i2v_first_last_pair_dasiwa_fast.json
```

Replace both the normal `LoadImage` input and the `Last Frame Image` input.

## 6. Practical defaults

- Use 48 GB VRAM or higher for the HQ/two-stage workflows.
- Keep `COMFYUI_ARGS=--reserve-vram 5` unless you know the GPU has room.
- If identity drifts, reduce prompt pressure and use the identity-lock workflow.
- If motion is weak, use the simple workflow and lower image-guide strength.
- Keep first-frame and last-frame images visually close for loop workflows.
