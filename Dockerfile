ARG BASE_IMAGE=runpod/comfyui:1.4.1-cuda12.8@sha256:1f9de5f6c1183211b7fb43c626c48a11c5ff9bda6acde77781fa1f104aac3469
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=120

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
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
    && python3 /opt/claude-ltx/scripts/check_workflow_nodes.py \
        --comfyui-dir /opt/comfyui-baked \
        --workflows /opt/claude-ltx/workflows

EXPOSE 8188

ENTRYPOINT []
CMD ["/opt/claude-ltx/scripts/start.sh"]
