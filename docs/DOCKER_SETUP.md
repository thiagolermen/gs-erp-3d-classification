# Docker Setup

> **Split workflow:**
> - **Local machine** — data download, preprocessing, notebooks (Python venv, CPU)
> - **Lab machine** — training and evaluation (Docker, GPU)

---

## Prerequisites (lab machine, one-time)

1. **Docker** ≥ 24.0 + Compose plugin
2. **NVIDIA Container Toolkit**:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
3. **NVIDIA driver** ≥ 525.60 (for CUDA 12.1)

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## First-time Setup

```bash
# 1. Clone the repo
git clone git@github.com:thiagolermen/gs-erp-3d-classification.git
cd gs-erp-3d-classification

# 2. Build the Docker image
make build

# 3. Verify GPU
make check-gpu
```

---

## Workflow

### Local machine — preprocess and cache ERPs

```bash
# Install deps (CPU-only PyTorch)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-local.txt
pip install -e . --no-deps

# Download ModelSplat (requires HuggingFace token)
python scripts/download_modelsplat.py --token <HF_TOKEN> --mn10-only
# Full MN40: python scripts/download_modelsplat.py --token <HF_TOKEN>

# Generate ERP cache (~8-shell, 512×256, stored in data/processed/)
bash scripts/preprocess_all.sh

# Transfer cache to lab machine
rsync -avz data/processed/ user@lab-machine:~/gs-erp-3d-classification/data/processed/
```

### Lab machine — train

```bash
# Open a tmux session so training survives SSH disconnects
tmux new-session -s training

# Train all 4 experiments sequentially
make baselines-all
# or individually:
make train CONFIG=configs/resnet34_hsdc_mn10.yaml

# Detach: Ctrl+B, D
# Reattach later: tmux attach -t training
```

### Evaluate

```bash
make evaluate \
  CONFIG=configs/resnet34_hsdc_mn10.yaml \
  CHECKPOINT=experiments/resnet34_hsdc_mn10_seed42/best_checkpoint.pt
```

---

## Training Outputs

```
experiments/<run_name>/
├── config.yaml         — copy of YAML config
├── train.log           — full logging output
├── metrics.csv         — per-epoch: loss, accuracy, lr
├── best_checkpoint.pt  — weights at best val accuracy
└── last_checkpoint.pt  — weights at final epoch
```

---

## Common Issues

**`could not select device driver "nvidia"`**
```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

**`Bus error` / DataLoader crash** — increase shared memory in `docker-compose.yml`:
```yaml
shm_size: '32g'
```

**`CUDA out of memory`** — reduce `batch_size` in the YAML config.

**`ModuleNotFoundError: No module named 'src'`** — outside Docker, set:
```bash
export PYTHONPATH=$(pwd)
```

---

## Jupyter (optional, remote access)

```bash
# Lab machine
make jupyter

# Home machine — open SSH tunnel
ssh -L 8888:localhost:8888 -N user@lab-machine

# Browser: http://localhost:8888
```

---

## CUDA Version Compatibility

The Dockerfile defaults to PyTorch 2.2 + CUDA 12.1. To build for a different driver:

```bash
docker compose build \
  --build-arg PYTORCH_VERSION=2.2.0 \
  --build-arg CUDA_VERSION=11.8 \
  --build-arg CUDNN_VERSION=8
```
