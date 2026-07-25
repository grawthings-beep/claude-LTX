#!/usr/bin/env bash
set -Eeuo pipefail

source /opt/claude-ltx/scripts/common.sh

COMFYUI_DIR="$(find_comfyui_dir)" || {
  echo "ERROR: could not find ComfyUI main.py. Set COMFYUI_DIR explicitly." >&2
  exit 2
}

PYTHON_BIN="$(find_python_bin)" || {
  echo "ERROR: neither python nor python3 was found in PATH." >&2
  exit 2
}

CUSTOM_NODES_DIR="${COMFYUI_DIR}/custom_nodes"
mkdir -p "${CUSTOM_NODES_DIR}"

while IFS='|' read -r name url ref; do
  [[ -z "${name}" || "${name}" =~ ^# ]] && continue
  target="${CUSTOM_NODES_DIR}/${name}"

  if [[ -d "${target}/.git" ]]; then
    echo "Updating custom node ${name}"
  else
    echo "Installing custom node ${name}"
    rm -rf "${target}"
    git init -q "${target}"
    git -C "${target}" remote add origin "${url}"
  fi

  if [[ -n "${ref:-}" ]]; then
    git -C "${target}" fetch --depth 1 origin "${ref}"
    git -C "${target}" checkout -q --detach FETCH_HEAD
  else
    git -C "${target}" fetch --depth 1 origin HEAD
    git -C "${target}" checkout -q --detach FETCH_HEAD
  fi

  if [[ -f "${target}/requirements.txt" ]]; then
    echo "Installing Python requirements for ${name}"
    "${PYTHON_BIN}" -m pip install -r "${target}/requirements.txt"
  fi

  rm -rf "${target}/.git" "${target}/.github" "${target}/tests"
  find "${target}" -type d -name __pycache__ -prune -exec rm -rf {} +
done < /opt/claude-ltx/custom_nodes.txt

install_bundled_nodepacks() {
  local source
  local target
  for source in /opt/claude-ltx/custom_nodepacks/*; do
    [[ -d "${source}" ]] || continue
    target="${CUSTOM_NODES_DIR}/$(basename "${source}")"
    echo "Installing bundled custom node $(basename "${source}")"
    mkdir -p "${target}"
    cp -a "${source}/." "${target}/"
  done
}

install_bundled_nodepacks
