ARG BASE_IMAGE=runpod/comfyui:1.4.1-cuda13.0@sha256:d49c7b0f8eb3fbf44725d210b2b0a86bdd8a972da7e32e5e73b0efafdebe3718
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

RUN rm -rf /opt/comfyui-baked/custom_nodes/ComfyUI-Manager \
    && chmod +x /opt/claude-ltx/scripts/*.sh \
    && /opt/claude-ltx/scripts/install_custom_nodes.sh \
    && python3 /opt/claude-ltx/scripts/check_workflow_nodes.py \
        --comfyui-dir /opt/comfyui-baked \
        --workflows /opt/claude-ltx/workflows

EXPOSE 8188

ENTRYPOINT []
CMD ["/opt/claude-ltx/scripts/start.sh"]
