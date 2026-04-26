# TropFormer cloud trainer — CUDA 12.1 base image
# ================================================
# Works on RunPod, Lambda Labs, vast.ai, AWS, GCP, Azure, your own server.
# Build:    docker build -t tropformer:cuda12 -f research/cloud/launch/Dockerfile .
# Run:      docker run --gpus all -v $(pwd):/workspace tropformer:cuda12 \
#               python -m research.cloud.train_cloud --preset medium_40gb --tag run0

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

LABEL maintainer="MK_III_Technologies"
LABEL project="TropFormer"
LABEL description="TSRNGist cloud trainer — auto-detect single GPU / DDP"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Minimal system deps; the pytorch base image already has python, cuda, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only the dependency files first for layer caching
COPY requirements.txt /workspace/requirements.txt
COPY research/cloud/requirements_cuda.txt /workspace/research/cloud/requirements_cuda.txt

# Base + CUDA-only deps
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt && \
    pip install -r /workspace/research/cloud/requirements_cuda.txt

# flash-attn must build against torch; install AFTER torch is in the image
RUN pip install --no-build-isolation flash-attn>=2.6 || \
    echo "flash-attn build skipped — image still functional via SDPA fallback"

# Now copy the actual source.  Doing this last lets us iterate on code without
# rebuilding all the dependency layers.
COPY . /workspace

# enwik8 will be downloaded into /workspace/data on first run (small, ~36 MB
# compressed).  For persistence across container runs, mount /workspace as a
# volume from the host or use a cloud volume (Modal/RunPod/etc.).
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/results /workspace/logs

# Default: print the help banner.  Override the CMD when running:
#   docker run --gpus all tropformer:cuda12 \
#       python -m research.cloud.train_cloud --preset medium_40gb --tag run0
CMD ["python", "-m", "research.cloud.train_cloud", "--help"]
