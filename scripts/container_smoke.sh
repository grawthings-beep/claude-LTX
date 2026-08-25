#!/usr/bin/env bash
set -Eeuo pipefail

SMOKE_ROOT="$(mktemp -d /tmp/claude-ltx-smoke.XXXXXX)"
COMFYUI_DIR="${COMFYUI_DIR:-/opt/comfyui-baked}"
EXTRA_YAML="${COMFYUI_DIR}/extra_model_paths.yaml"
EXTRA_YML="${COMFYUI_DIR}/extra_model_paths.yml"

[[ -f "${EXTRA_YAML}" ]] && cp "${EXTRA_YAML}" "${SMOKE_ROOT}/extra_model_paths.yaml.bak"
[[ -f "${EXTRA_YML}" ]] && cp "${EXTRA_YML}" "${SMOKE_ROOT}/extra_model_paths.yml.bak"

cleanup() {
  if [[ -f "${SMOKE_ROOT}/extra_model_paths.yaml.bak" ]]; then
    cp "${SMOKE_ROOT}/extra_model_paths.yaml.bak" "${EXTRA_YAML}"
  else
    rm -f -- "${EXTRA_YAML}"
  fi
  if [[ -f "${SMOKE_ROOT}/extra_model_paths.yml.bak" ]]; then
    cp "${SMOKE_ROOT}/extra_model_paths.yml.bak" "${EXTRA_YML}"
  else
    rm -f -- "${EXTRA_YML}"
  fi
  rm -rf -- "${SMOKE_ROOT}"
}
trap cleanup EXIT

mkdir -p "${SMOKE_ROOT}/workspace/models/auto_mosaic"
printf 'container-smoke-placeholder\n' > \
  "${SMOKE_ROOT}/workspace/models/auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"

env \
  DOWNLOAD_MODELS=0 \
  RUN_DEP_CHECK=0 \
  WORKSPACE_DIR="${SMOKE_ROOT}/workspace" \
  MODEL_ROOT="${SMOKE_ROOT}/workspace" \
  CONFIG_DIR="${SMOKE_ROOT}/config" \
  COMFYUI_DIR="${COMFYUI_DIR}" \
  COMFYUI_ARGS="--cpu --quick-test-for-ci --max-upload-size 300" \
  /opt/claude-ltx/scripts/start.sh

test -d "${SMOKE_ROOT}/workspace/user/default/workflows"
test "$(find "${SMOKE_ROOT}/workspace/user/default/workflows" -maxdepth 1 -name '*.json' | wc -l)" -ge 3
