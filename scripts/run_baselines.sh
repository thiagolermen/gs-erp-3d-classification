#!/usr/bin/env bash
# =============================================================================
# run_baselines.sh — Train all four baseline experiments sequentially.
#
# Run INSIDE the Docker container:
#   bash scripts/run_baselines.sh
#
# Or from the host via make:
#   make baselines-all
#
# Baseline targets (from ai-developer.md):
#   1. ResNet-34 + HSDC  on ModelNet10  → target 97.1%
#   2. ResNet-34 + HSDC  on ModelNet40  → target 93.9%
#   3. ResNet-50 + SWHDC on ModelNet10  → target 94.11%
#   4. ResNet-50 + SWHDC on ModelNet40  → target 91.89%
#
# Outputs written to experiments/<run_name>/ (logs, checkpoints, metrics.csv).
# =============================================================================

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIGS=(
    "configs/resnet34_hsdc_mn10.yaml"
    "configs/resnet34_hsdc_mn40.yaml"
    "configs/resnet50_swhdc_mn10.yaml"
    "configs/resnet50_swhdc_mn40.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    log "──────────────────────────────────────────────────"
    log "Training: $cfg"
    log "──────────────────────────────────────────────────"
    python -m src.training.train --config "$cfg"
    log "Finished: $cfg"
done

log "All baseline experiments complete."
log "Results in experiments/:"
ls -1 experiments/ 2>/dev/null || true
