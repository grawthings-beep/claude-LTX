ARG BASE_IMAGE=runpod/comfyui:1.4.1-cuda12.8
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
        ca-certificates \
        curl \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade \
        "huggingface_hub>=0.32.0,<1.0" \
        "hf_xet>=1.1.0"

COPY custom_nodes.txt /opt/claude-ltx/custom_nodes.txt
COPY config/ /opt/claude-ltx/config/
COPY scripts/ /opt/claude-ltx/scripts/
COPY workflows/ /opt/claude-ltx/workflows/

RUN chmod +x /opt/claude-ltx/scripts/*.sh \
    && /opt/claude-ltx/scripts/install_custom_nodes.sh

EXPOSE 8188

ENTRYPOINT []
CMD ["/opt/claude-ltx/scripts/start.sh"]
