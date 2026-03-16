# Docker Setup

> **Split workflow:**
> - **Local machine** — data download, preprocessing, notebooks (Python venv, CPU)
> - **Lab machine (`lermen@anubis`)** — training and evaluation (Docker, GPU)
>
> Lab machine specs: Ubuntu 22.04.1 LTS (Linux 6.8.0-101-generic x86_64), Docker 26.0.0

---

## Prerequisites (lab machine, one-time)

Docker 26.0.0 is already installed on `anubis`. Only the NVIDIA Container Toolkit needs to be configured if not done yet:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

NVIDIA driver requirement: ≥ 525.60 (for CUDA 12.1).

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## First-time Setup on anubis

```bash
# 1. Clone the repo (already done at ~/DEV_ENV/gs-erp-3d-classification)
git clone https://github.com/thiagolermen/gs-erp-3d-classification.git
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

# Transfer cache to anubis
rsync -avz data/processed/ lermen@anubis:~/DEV_ENV/gs-erp-3d-classification/data/processed/
```

### Lab machine (anubis) — train

```bash
# SSH into anubis
ssh lermen@anubis

cd ~/DEV_ENV/gs-erp-3d-classification

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
# Lab machine (anubis)
make jupyter

# Home machine — open SSH tunnel
ssh -L 8888:localhost:8888 -N lermen@anubis

# Browser: http://localhost:8888   token: erp-vit
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
