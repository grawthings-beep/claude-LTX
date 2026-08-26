ARG BASE_IMAGE=runpod/comfyui:1.4.4-cuda12.8@sha256:7078f94dbe28d079c487c245dc3524443e2c6225a6208a1fff8c7a652c1b3a40
FROM ${BASE_IMAGE}

ARG BUILD_REVISION=unknown
ARG CUDA_VARIANT=12.8

LABEL org.opencontainers.image.revision="${BUILD_REVISION}" \
      io.runpod.cuda.variant="${CUDA_VARIANT}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    CUDA_FORCE_PRELOAD_LIBRARIES=0 \
    CLAUDE_LTX_REVISION="${BUILD_REVISION}" \
    CLAUDE_LTX_CUDA_VARIANT="${CUDA_VARIANT}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        nvidia-modprobe \
    && rm -rf /var/lib/apt/lists/*

COPY custom_nodes.txt /opt/claude-ltx/custom_nodes.txt
COPY custom_nodepacks/ /opt/claude-ltx/custom_nodepacks/
COPY config/ /opt/claude-ltx/config/
COPY scripts/ /opt/claude-ltx/scripts/
COPY workflows/ /opt/claude-ltx/workflows/

RUN find /opt/comfyui-baked/custom_nodes -mindepth 1 -maxdepth 1 \
        -exec rm -rf {} + \
    && chmod +x /opt/claude-ltx/scripts/*.sh \
    && /opt/claude-ltx/scripts/install_custom_nodes.sh \
    && python3 -c "import ultralytics; assert ultralytics.__version__ == '8.4.104'" \
    && python3 /opt/claude-ltx/scripts/check_workflow_nodes.py \
        --comfyui-dir /opt/comfyui-baked \
        --workflows /opt/claude-ltx/workflows \
    && /opt/claude-ltx/scripts/container_smoke.sh

EXPOSE 8188

ENTRYPOINT []
CMD ["/opt/claude-ltx/scripts/start.sh"]
