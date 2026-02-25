# Docker Setup Guide

> Complete instructions for running the ERP-ViT pipeline on a remote Linux GPU machine accessed via SSH.

---

## Prerequisites

### On the lab machine (Linux, one-time setup)

1. **Docker Engine** ≥ 24.0 with the Compose plugin:
   ```bash
   # Verify
   docker --version        # Docker version 24+
   docker compose version  # Docker Compose version v2+
   ```

2. **NVIDIA Container Toolkit** (for GPU access inside containers):
   ```bash
   # Install (Ubuntu/Debian)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker

   # Verify GPU is visible inside Docker
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **NVIDIA driver** ≥ 525.60.13 (for CUDA 12.1):
   ```bash
   nvidia-smi   # check driver version
   ```

### CUDA version compatibility

The `Dockerfile` defaults to `PyTorch 2.2.0 + CUDA 12.1`. If the lab machine has a different driver, rebuild with a compatible base:

| Driver version | Max CUDA | Recommended build arg |
|---|---|---|
| ≥ 525.60 | CUDA 12.x | `CUDA_VERSION=12.1` (default) |
| 520–524 | CUDA 11.8 | `CUDA_VERSION=11.8` |
| 450–519 | CUDA 11.3 | `CUDA_VERSION=11.3` |

```bash
# Build for CUDA 11.8
docker compose build \
  --build-arg PYTORCH_VERSION=2.2.0 \
  --build-arg CUDA_VERSION=11.8 \
  --build-arg CUDNN_VERSION=8
```

---

## First-time Setup

```bash
# 1. Clone the repository on the lab machine
git clone <repo-url> erp-vit-3d-classification
cd erp-vit-3d-classification

# 2. Create the data directories (mounted as Docker volumes)
mkdir -p data/raw/modelnet10 data/raw/modelnet40
mkdir -p experiments notebooks

# 3. Download ModelNet (Princeton 3D Shape Benchmark)
#    https://modelnet.cs.princeton.edu/
#    Extract so the directory structure is:
#      data/raw/modelnet10/<class>/train/*.off
#      data/raw/modelnet10/<class>/test/*.off
#      data/raw/modelnet40/<class>/train/*.off
#      data/raw/modelnet40/<class>/test/*.off

# 4. Build the Docker image
make build
# or: docker compose build

# 5. Verify GPU access
make check-gpu
```

Expected output from `make check-gpu`:
```
  Python       : 3.10.x
  PyTorch      : 2.2.0
  CUDA         : available ✓
  CUDA version : 12.1
  GPU count    : 1
  GPU 0        : NVIDIA GeForce RTX 3090  |  24,576 MB total  |  24,100 MB free
  Tensor test  : OK ✓
  All checks passed — ready to train.
```

---

## Preprocessing (run once per dataset)

ERP images are pre-computed from raw `.off` meshes and cached as `.npy` files. This is the most time-consuming step (~2–10 s per mesh) and only needs to run once.

```bash
# Estimated times (single-core CPU inside container)
#   ModelNet10 HSDC  : ~2 h   →  ~3.1 GB cache
#   ModelNet10 SWHDC : ~1 h   →  ~0.3 GB cache
#   ModelNet40 HSDC  : ~4 h   →  ~7.6 GB cache
#   ModelNet40 SWHDC : ~2 h   →  ~0.6 GB cache

# All four (sequential, ~9 h total)
make preprocess-all

# Or one at a time (e.g. start with baselines)
make preprocess-mn10-hsdc
make preprocess-mn10-swhdc
```

The cache is stored at `data/processed/<dataset>/<pipeline>/` and is persistent across container restarts (Docker volume mount).

Preprocessing is **idempotent** — if interrupted, re-running it skips already-cached files.

---

## Training

### Interactive (foreground)

Suitable for short runs or debugging. The training loop prints live progress to the terminal.

```bash
# Baseline 1: ResNet-34 + HSDC on ModelNet10 (target: 97.1%)
make train CONFIG=configs/resnet34_hsdc_mn10.yaml

# Proposed: Swin-T + HSDC on ModelNet40
make train CONFIG=configs/swin_hsdc_mn40.yaml
```

### Long-running (background via tmux)

For multi-hour/multi-day experiments, use `tmux` to keep the session alive after disconnecting:

```bash
# On the lab machine
tmux new-session -s training        # create a new session named "training"

make train CONFIG=configs/resnet34_hsdc_mn40.yaml
# ...training runs...

# Detach from tmux (does NOT stop the training): Ctrl+B, then D

# From home (SSH):
ssh user@lab-machine
tmux attach -t training             # reattach to see live progress

# Or watch the log file directly (no tmux needed):
make logs RUN_NAME=resnet34_hsdc_mn40_seed42
```

### Running all baselines sequentially

```bash
# Inside a tmux session:
bash scripts/run_baselines.sh

# Or via make:
make baselines-all
```

### Training outputs

Every experiment writes to `experiments/<run_name>/`:

```
experiments/resnet34_hsdc_mn10_seed42/
├── config.yaml           — copy of the YAML config used
├── train.log             — full Python logging output
├── metrics.csv           — epoch, train_loss, val_loss, train_acc, val_acc, lr
├── best_checkpoint.pt    — model weights at best validation accuracy
└── last_checkpoint.pt    — model weights at final epoch
```

---

## Evaluation

```bash
make evaluate \
  CONFIG=configs/resnet34_hsdc_mn10.yaml \
  CHECKPOINT=experiments/resnet34_hsdc_mn10_seed42/best_checkpoint.pt
```

Reports top-1 accuracy, per-class accuracy, and a full sklearn classification report.

---

## Jupyter Notebook (remote access from home)

### Step 1 — Start Jupyter on the lab machine

```bash
make jupyter
```

This starts Jupyter on port 8888 inside the container, accessible from the host.

### Step 2 — SSH tunnel from home

Open a new terminal on your home machine and run:

```bash
ssh -L 8888:localhost:8888 -N user@lab-machine
# -L: forward local port 8888 → remote port 8888
# -N: no command, tunnel only
# Keep this terminal open while using Jupyter
```

### Step 3 — Open in browser

```
http://localhost:8888
Token: erp-vit
```

You now have a full Jupyter environment with direct access to `experiments/`, `data/`, and `notebooks/`.

---

## GPU monitoring

From the lab machine (outside the container):

```bash
# Real-time GPU utilisation
watch -n 1 nvidia-smi

# GPU memory for all processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Inside the container (e.g. via make shell):
python -c "import torch; print(torch.cuda.memory_summary())"
```

---

## Multi-GPU training

The training script automatically wraps the model in `torch.nn.DataParallel` when multiple GPUs are detected. No configuration change is needed.

To restrict training to specific GPUs, set `NVIDIA_VISIBLE_DEVICES` in `docker-compose.yml`:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0,1   # use GPUs 0 and 1 only
```

---

## Common issues

### `could not select device driver "nvidia"`

```bash
# nvidia-container-toolkit is not configured
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### `Bus error` or DataLoader crashes

Shared memory too small. Increase `shm_size` in `docker-compose.yml`:

```yaml
shm_size: '32g'
```

Or use `ipc: host` (shares the host IPC namespace, less isolated):

```yaml
ipc: host
# remove the shm_size line
```

### `CUDA out of memory`

Reduce batch size in the YAML config:
```yaml
training:
  batch_size: 16   # default is 32
```

### Preprocessing interrupted mid-run

Safe to restart — the script skips `.npy` files that already exist.

### `ModuleNotFoundError: No module named 'src'`

The container sets `PYTHONPATH=/workspace` automatically. If running outside Docker, export it manually:

```bash
export PYTHONPATH=$(pwd)
```

---

## Typical workflow summary

```
Lab machine                         Home machine
──────────────────────────────────  ──────────────────
1. git clone + data setup
2. make build
3. make check-gpu
4. make preprocess-all              (wait ~9 h)
5. tmux new -s exp1
6. make baselines-all               Ctrl+B, D to detach
                                    ssh -L 8888:localhost:8888 user@lab
                                    make jupyter  (from lab)
                                    http://localhost:8888  (from browser)
7. (after baselines pass)
   make train CONFIG=configs/swin_hsdc_mn40.yaml
8. make evaluate CONFIG=... CHECKPOINT=...
```
