# =============================================================================
# GS-ERP 3D Classification — Docker Image
# =============================================================================
#
# Base:  PyTorch 2.2.0 + CUDA 12.1 + cuDNN 8 (runtime)
# Host requirements:
#   - NVIDIA driver >= 525.60.13  (for CUDA 12.1)
#   - Lab GPU: NVIDIA GeForce RTX 2070 (8 GB VRAM), driver 560.35.03 (CUDA 12.6 capable)
#     CUDA 12.1 image is fully compatible with driver 560.35.03
#   - nvidia-container-toolkit installed and configured
#
# Build:
#   docker compose build
#
# Quick start:
#   docker compose run --rm gs-erp bash
# =============================================================================

# Allow overriding CUDA/PyTorch version at build time
ARG PYTORCH_VERSION=2.2.0
ARG CUDA_VERSION=12.1
ARG CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

LABEL org.opencontainers.image.title="gs-erp-3d-classification" \
      org.opencontainers.image.description="GS-ERP 3D object classification (UFRGS TCC)" \
      org.opencontainers.image.source="https://github.com/thiagolermen/gs-erp-3d-classification"

# =============================================================================
# 1. System dependencies
# =============================================================================
# libgl1-mesa-glx + libglib2.0-0 : required by OpenCV / trimesh rendering
# libsm6 + libxrender1 + libxext6 : headless OpenGL / X11 stubs
# libgomp1                        : OpenMP (used by trimesh / numpy)
# git + wget                      : utility tools
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# 2. Python dependencies
# =============================================================================
# Copy only the dependency files first so Docker can cache this layer.
# Re-run only when requirements.txt or pyproject.toml changes.
# =============================================================================
WORKDIR /workspace

COPY requirements.txt pyproject.toml ./

# torch / torchvision are already in the base image at the exact versions
# listed in requirements.txt; pip will skip reinstalling them.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# =============================================================================
# 3. Project source
# =============================================================================
COPY src/         ./src/
COPY configs/     ./configs/
COPY tests/       ./tests/
COPY notebooks/   ./notebooks/

# Install the project package in editable mode so that
#   `from src.models import ...` works from any working directory.
# --no-deps: skip re-resolving the already-installed dependencies.
RUN pip install --no-cache-dir -e . --no-deps

# =============================================================================
# 4. Environment variables
# =============================================================================
# PYTHONPATH: fallback for scripts run outside the installed package
ENV PYTHONPATH=/workspace

# Headless matplotlib — no display server required in training/eval jobs
ENV MPLBACKEND=Agg

# Limit intra-op parallelism; DataLoader workers provide data-level parallelism
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Suppress tokenizer parallelism warnings from HuggingFace (used transitively by timm)
ENV TOKENIZERS_PARALLELISM=false

# =============================================================================
# 5. Runtime directories
# =============================================================================
# Create mount-point stubs; actual content comes from docker-compose volumes.
RUN mkdir -p /workspace/gs_data \
             /workspace/data/processed \
             /workspace/experiments \
             /workspace/notebooks

# =============================================================================
# 6. Jupyter configuration
# =============================================================================
# Configure Jupyter Server for SSH-tunnel access:
#   - Listen on all interfaces (required when running inside a container)
#   - Disable browser auto-launch (headless server)
#   - Set a fixed token to avoid re-entering it after restart
#   - Allow root (the container runs as root by default)
#
# Access from home via SSH tunnel:
#   ssh -L 8888:localhost:8888 lermen@anubis
#   # then open http://localhost:8888 with token 'erp-vit'
# =============================================================================
RUN jupyter server --generate-config 2>/dev/null || \
    jupyter notebook --generate-config 2>/dev/null || true

RUN python - <<'PYEOF'
import os, pathlib

# Try jupyter_server_config.py (newer), fall back to jupyter_notebook_config.py
candidates = [
    pathlib.Path.home() / ".jupyter" / "jupyter_server_config.py",
    pathlib.Path.home() / ".jupyter" / "jupyter_notebook_config.py",
]
lines = [
    "c.ServerApp.ip = '0.0.0.0'\n",
    "c.ServerApp.port = 8888\n",
    "c.ServerApp.open_browser = False\n",
    "c.ServerApp.allow_root = True\n",
    "c.ServerApp.token = 'erp-vit'\n",
    "c.ServerApp.password = ''\n",
    # Classic NotebookApp aliases (backwards-compatible)
    "c.NotebookApp.ip = '0.0.0.0'\n",
    "c.NotebookApp.port = 8888\n",
    "c.NotebookApp.open_browser = False\n",
    "c.NotebookApp.allow_root = True\n",
    "c.NotebookApp.token = 'erp-vit'\n",
    "c.NotebookApp.password = ''\n",
]
for cfg in candidates:
    if cfg.exists():
        cfg.write_text("".join(lines))
        break
PYEOF

EXPOSE 8888

# =============================================================================
# Default command: interactive bash shell
# Override with: docker compose run erp-vit python -m src.training.train ...
# =============================================================================
CMD ["bash"]
