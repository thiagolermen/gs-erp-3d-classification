# Docker Setup

> **Split workflow:**
> - **Local machine** — data download, preprocessing, notebooks (Python venv, CPU)
> - **Lab machine (`lermen@anubis`)** — training and evaluation (Docker, GPU)
>
> Lab machine specs: Ubuntu 22.04.1 LTS (Linux 6.8.0-101-generic x86_64), Docker 26.0.0
> GPU: NVIDIA GeForce RTX 2070 — 8 GB VRAM, driver 560.35.03, CUDA 12.6 capable

---

## Prerequisites (lab machine, one-time)

Docker 26.0.0 is already installed on `anubis`. Driver 560.35.03 supports up to CUDA 12.6 and is fully compatible with the CUDA 12.1 container image. Only the NVIDIA Container Toolkit needs to be configured if not done yet:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Expected: RTX 2070 listed, CUDA Version 12.6
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

## Data Preprocessing (inside Docker on anubis)

The full preprocessing pipeline runs inside the container. It has three stages:

```
HuggingFace  →  .zip download  →  extract PLY files  →  radiance field ERP cache (.npy)
```

### Stage 1 — Download the ModelSplat dataset

The ModelSplat dataset (ShapeSplats/ModelNet_Splats) provides pre-trained 3DGS `.ply` files for every ModelNet object. A HuggingFace access token is required.

```bash
# ModelNet10 only (~15 GB — recommended for initial runs)
make download TOKEN=<HF_TOKEN> MN10_ONLY=1

# Full ModelNet40 (~40 GB)
make download TOKEN=<HF_TOKEN>
```

The script downloads per-category zip files, extracts them, and prints a summary. Extracted PLY files land at:

```
gs_data/modelsplat/modelsplat_ply/<category>/train|test/<id>/point_cloud.ply
```

Already-downloaded or already-extracted entries are skipped automatically, so the command is safe to re-run after an interruption.

You can also download specific categories:

```bash
make shell
# inside the container:
python scripts/download_modelsplat.py --token <HF_TOKEN> --categories sofa chair table
```

### Stage 2 — (No mesh generation needed)

The ModelSplat dataset ships with pre-trained 3DGS `.ply` files — there is no intermediate mesh or PLY generation step. The downloaded `point_cloud.ply` files are the 3D Gaussian Splat representations used directly in Stage 3.

### Stage 3 — Generate the radiance field ERP cache

This step reads each `point_cloud.ply`, places a virtual camera at the opacity-weighted centroid, samples the radiance field at 8 concentric spherical shells (EgoNeRF exponential spacing), and writes an `(8, 256, 512)` float32 `.npy` file to `data/processed/`.

```bash
# Preprocess ModelNet10 (recommended first)
make preprocess DATASET=mn10

# Preprocess ModelNet40
make preprocess DATASET=mn40

# Both in sequence
make preprocess DATASET=all
```

Or run manually inside the container for full control:

```bash
make shell
# inside the container:
python -m src.preprocessing.dataset \
    --data_root  gs_data/modelsplat/modelsplat_ply \
    --cache_dir  data/processed/modelnet10/radiance_field \
    --pipeline   radiance_field \
    --dataset    modelnet10 \
    --n_shells   8 \
    --erp_height 256 \
    --erp_width  512
```

Cache outputs:

```
data/processed/
├── modelnet10/radiance_field/<category>/train|test/<id>.npy   # shape (8, 256, 512)
└── modelnet40/radiance_field/<category>/train|test/<id>.npy
```

Existing `.npy` files are skipped, so preprocessing can be safely resumed after interruption.

> **Tip:** Run preprocessing inside a `tmux` session — it can take several hours for MN40:
> ```bash
> tmux new-session -s preprocess
> make preprocess DATASET=all
> # Ctrl+B, D to detach; tmux attach -t preprocess to reattach
> ```

### Full pipeline (all-in-one)

```bash
# 1. Enter a persistent tmux session
tmux new-session -s pipeline

# 2. Download MN10 PLY files
make download TOKEN=<HF_TOKEN> MN10_ONLY=1

# 3. Generate ERP cache for MN10
make preprocess DATASET=mn10

# 4. Verify a sample ERP was written
ls data/processed/modelnet10/radiance_field/

# 5. Proceed to training
make train CONFIG=configs/resnet34_hsdc_mn10.yaml
```

---

## Workflow

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

# Detach: Ctrl+B, D   |   Reattach: tmux attach -t training
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

**`CUDA out of memory`** — The RTX 2070 has 8 GB VRAM. All configs use `mixed_precision: true` and `batch_size: 32`, which fits comfortably for ResNet-34+HSDC (~5.3 M params). ResNet-50+SWHDC (~25.5 M params) may be tighter; if OOM occurs reduce `batch_size` to 16 in the YAML config:
```yaml
data:
  batch_size: 16
```

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
