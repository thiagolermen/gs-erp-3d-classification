#!/usr/bin/env bash
# =============================================================================
# preprocess_all.sh — Generate ERP caches for all dataset / pipeline combos.
#
# Run INSIDE the Docker container:
#   bash scripts/preprocess_all.sh
#
# Or from the host via make:
#   make preprocess-all
#
# Prerequisites:
#   data/raw/modelnet10/  — extracted ModelNet10 .off files
#   data/raw/modelnet40/  — extracted ModelNet40 .off files
#
# Outputs (appended incrementally; existing .npy files are skipped):
#   data/processed/modelnet10/hsdc/   — 12-ch (C, 256, 512) float32 arrays
#   data/processed/modelnet10/swhdc/  —  1-ch (C, 256, 512) float32 arrays
#   data/processed/modelnet40/hsdc/
#   data/processed/modelnet40/swhdc/
# =============================================================================

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Check raw data exists ────────────────────────────────────────────────────
for dataset in modelnet10 modelnet40; do
    if [ ! -d "data/raw/$dataset" ]; then
        echo "ERROR: data/raw/$dataset not found."
        echo "Download ModelNet from https://modelnet.cs.princeton.edu/ and"
        echo "extract into data/raw/modelnet10/ and data/raw/modelnet40/."
        exit 1
    fi
done

# ── Run preprocessing ────────────────────────────────────────────────────────
JOBS=(
  "modelnet10 hsdc"
  "modelnet10 swhdc"
  "modelnet40 hsdc"
  "modelnet40 swhdc"
)

for job in "${JOBS[@]}"; do
    dataset=$(echo "$job" | cut -d' ' -f1)
    pipeline=$(echo "$job" | cut -d' ' -f2)
    log "Starting: $dataset / $pipeline"
    python -m src.preprocessing.dataset \
        --data_root "data/raw/$dataset" \
        --cache_dir "data/processed/$dataset/$pipeline" \
        --pipeline  "$pipeline"
    log "Done:     $dataset / $pipeline"
done

log "All ERP caches generated."
