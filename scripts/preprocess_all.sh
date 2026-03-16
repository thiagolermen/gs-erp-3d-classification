#!/usr/bin/env bash
# =============================================================================
# preprocess_all.sh — Generate radiance field ERP caches from 3DGS PLY files.
#
# Run INSIDE the Docker container:
#   bash scripts/preprocess_all.sh
#
# Or from the host via make:
#   make preprocess-all
#
# Prerequisites:
#   gs_data/modelsplat/modelsplat_ply/  — extracted ModelSplat PLY directories
#   Download via: python scripts/download_modelsplat.py --token <HF_TOKEN> --mn10-only
#
# Outputs (appended incrementally; existing cache entries are skipped):
#   data/processed/modelnet10/radiance_field/  — 8-ch (N_shells, H, W) float32
#   data/processed/modelnet40/radiance_field/  — 8-ch (N_shells, H, W) float32
# =============================================================================

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GS_ROOT="${GS_ROOT:-gs_data/modelsplat/modelsplat_ply}"

# ── Check PLY data exists ─────────────────────────────────────────────────────
if [ ! -d "$GS_ROOT" ]; then
    echo "ERROR: $GS_ROOT not found."
    echo "Download ModelSplat dataset first:"
    echo "  python scripts/download_modelsplat.py --token <HF_TOKEN> --mn10-only"
    exit 1
fi

# ── Run preprocessing ─────────────────────────────────────────────────────────
DATASETS=("modelnet10" "modelnet40")

for dataset in "${DATASETS[@]}"; do
    log "Starting: $dataset / radiance_field"
    python -m src.preprocessing.dataset \
        --data_root    "$GS_ROOT" \
        --cache_dir    "data/processed/$dataset/radiance_field" \
        --pipeline     radiance_field \
        --dataset      "$dataset" \
        --n_shells     8 \
        --erp_height   256 \
        --erp_width    512
    log "Done:     $dataset / radiance_field"
done

log "All ERP caches generated."
