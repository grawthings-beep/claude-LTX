ARG BASE_IMAGE=runpod/comfyui:1.4.2-cuda12.8@sha256:7165fc8867ff4e8ffdcfc328cddfb2ad007493cad62e42a90054a7a5cdb7eec8
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

RUN python3 -m pip install --upgrade \
        "huggingface_hub>=0.32.0,<1.0" \
        "hf_xet>=1.1.0"

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
