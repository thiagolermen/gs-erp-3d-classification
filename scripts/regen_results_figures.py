"""Regenerate the data-driven results figures from the stratified runs.

Reads the eight runs under ``experiments/last_experiment/experiments`` (four
distortion-correction block variants and four plain-ResNet ablation baselines)
and rebuilds the figures referenced by the Results chapter of the TCC:

    - confusion_matrix_<run>.pdf      (four block variants)
    - training_curves_combined.pdf    (eight runs, grouped by benchmark)
    - pareto_mn10.pdf / pareto_mn40.pdf (RF-ERP points + literature)
    - per_class_comparison_mn10.pdf / _mn40.pdf  (block vs. its baseline)

Outputs land in ``experiments/figures`` and are copied into
``tcc/images/results`` so the LaTeX picks them up unchanged.

Run:  python -m scripts.regen_results_figures
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.metrics import (
    MODELNET10_CLASSES,
    MODELNET40_CLASSES,
    ema_smooth,
)
from src.analysis.visualize import plot_confusion_matrix, plot_pareto, _save

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "experiments" / "last_experiment" / "experiments"
FIG_DIR = ROOT / "experiments" / "figures"
TCC_DIR = ROOT / "tcc" / "images" / "results"

# ---------------------------------------------------------------------------
# Run registry: name -> (benchmark, backbone-label, is_block, colour)
# ---------------------------------------------------------------------------
RUNS = {
    "resnet34_baseline_mn10_seed42": ("MN10", "ResNet-34 (plain)", False, "#7f7f7f"),
    "resnet34_hsdc_mn10_seed42":     ("MN10", "ResNet-34 + HSDC",  True,  "#2166ac"),
    "resnet50_baseline_mn10_seed42": ("MN10", "ResNet-50 (plain)", False, "#e08214"),
    "resnet50_swhdc_mn10_seed42":    ("MN10", "ResNet-50 + SWHDC", True,  "#d6604d"),
    "resnet34_baseline_mn40_seed42": ("MN40", "ResNet-34 (plain)", False, "#7f7f7f"),
    "resnet34_hsdc_mn40_seed42":     ("MN40", "ResNet-34 + HSDC",  True,  "#2166ac"),
    "resnet50_baseline_mn40_seed42": ("MN40", "ResNet-50 (plain)", False, "#e08214"),
    "resnet50_swhdc_mn40_seed42":    ("MN40", "ResNet-50 + SWHDC", True,  "#d6604d"),
}


def _classes(benchmark: str):
    return MODELNET10_CLASSES if benchmark == "MN10" else MODELNET40_CLASSES


def _per_class_recall(run: str) -> np.ndarray:
    cm = np.load(RUNS_DIR / run / "confusion_matrix.npy").astype(float)
    return np.diag(cm) / cm.sum(axis=1).clip(min=1)


def _save_both(fig: plt.Figure, stem: str) -> None:
    """Save under experiments/figures and copy the PDF into tcc/images/results."""
    _save(fig, FIG_DIR / stem)
    plt.close(fig)
    src = FIG_DIR / f"{stem}.pdf"
    if src.exists():
        TCC_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, TCC_DIR / f"{stem}.pdf")


# ---------------------------------------------------------------------------
# 1. Confusion matrices (four block variants)
# ---------------------------------------------------------------------------
def make_confusion_matrices() -> None:
    for run, (bench, label, is_block, _) in RUNS.items():
        if not is_block:
            continue
        cm = np.load(RUNS_DIR / run / "confusion_matrix.npy")
        fig = plot_confusion_matrix(cm, _classes(bench), title=f"{label} — {bench}")
        _save_both(fig, f"confusion_matrix_{run}")
        print(f"  confusion_matrix_{run}")


# ---------------------------------------------------------------------------
# 2. Combined training curves (8 runs, grouped by benchmark)
# ---------------------------------------------------------------------------
def make_training_curves() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for r, bench in enumerate(("MN10", "MN40")):
        ax_loss, ax_acc = axes[r, 0], axes[r, 1]
        for run, (b, label, _is_block, colour) in RUNS.items():
            if b != bench:
                continue
            df = pd.read_csv(RUNS_DIR / run / "metrics.csv")
            ep = df["epoch"].values
            # training curves: faded; validation curves: solid
            ax_loss.plot(ep, ema_smooth(df["train_loss"].values), color=colour,
                         alpha=0.25, linewidth=1.0)
            ax_loss.plot(ep, ema_smooth(df["val_loss"].values), color=colour,
                         linewidth=1.6, label=label)
            ax_acc.plot(ep, ema_smooth(df["train_acc"].values), color=colour,
                        alpha=0.25, linewidth=1.0)
            ax_acc.plot(ep, ema_smooth(df["val_acc"].values), color=colour,
                        linewidth=1.6, label=label)
        ax_loss.set_title(f"{bench} — loss")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
        ax_acc.set_title(f"{bench} — accuracy")
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(40, 100)
        ax_acc.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save_both(fig, "training_curves_combined")
    print("  training_curves_combined")


# ---------------------------------------------------------------------------
# 3. Pareto plots (RF-ERP points + literature)
# ---------------------------------------------------------------------------
def make_pareto() -> None:
    # proposed RF-ERP points: label -> (params_M, test_oa_pct)
    proposed = {
        "MN10": {
            "ResNet-34+HSDC (ours)":  (5.47, 90.75),
            "ResNet-50+SWHDC (ours)": (23.55, 89.54),
            "ResNet-34 plain (ours)": (21.31, 89.76),
            "ResNet-50 plain (ours)": (23.55, 91.52),
        },
        "MN40": {
            "ResNet-34+HSDC (ours)":  (5.49, 86.38),
            "ResNet-50+SWHDC (ours)": (23.61, 87.11),
            "ResNet-34 plain (ours)": (21.33, 88.37),
            "ResNet-50 plain (ours)": (23.61, 88.73),
        },
    }
    for bench in ("MN10", "MN40"):
        fig = plot_pareto(proposed[bench], dataset=bench)
        _save_both(fig, f"pareto_{bench.lower()}")
        print(f"  pareto_{bench.lower()}")


# ---------------------------------------------------------------------------
# 4. Per-class comparison: each block vs. its plain baseline
# ---------------------------------------------------------------------------
_PAIRS = {
    "MN10": [
        ("ResNet-34: HSDC vs plain", "resnet34_hsdc_mn10_seed42",
         "resnet34_baseline_mn10_seed42"),
        ("ResNet-50: SWHDC vs plain", "resnet50_swhdc_mn10_seed42",
         "resnet50_baseline_mn10_seed42"),
    ],
    "MN40": [
        ("ResNet-34: HSDC vs plain", "resnet34_hsdc_mn40_seed42",
         "resnet34_baseline_mn40_seed42"),
        ("ResNet-50: SWHDC vs plain", "resnet50_swhdc_mn40_seed42",
         "resnet50_baseline_mn40_seed42"),
    ],
}


def make_per_class_comparison() -> None:
    for bench, pairs in _PAIRS.items():
        classes = np.array(_classes(bench))
        n = len(classes)
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, n * 0.34)))
        for ax, (title, block_run, base_run) in zip(axes, pairs):
            block = _per_class_recall(block_run) * 100
            base = _per_class_recall(base_run) * 100
            order = np.argsort(block)  # worst block class at top
            y = np.arange(n)
            h = 0.4
            ax.barh(y + h / 2, base[order], height=h, color="#7f7f7f",
                    alpha=0.85, label="plain ResNet")
            ax.barh(y - h / 2, block[order], height=h, color="#2166ac",
                    alpha=0.9, label="+ block")
            ax.set_yticks(y)
            ax.set_yticklabels(classes[order], fontsize=7)
            ax.set_xlim(0, 105)
            ax.set_xlabel("Per-class recall (%)")
            ax.set_title(title)
            ax.legend(loc="lower right", fontsize=8)
        fig.suptitle(f"Per-class recall: distortion-correction block "
                     f"vs. plain baseline on {bench}", y=1.005)
        fig.tight_layout()
        _save_both(fig, f"per_class_comparison_{bench.lower()}")
        print(f"  per_class_comparison_{bench.lower()}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Regenerating results figures from", RUNS_DIR)
    make_confusion_matrices()
    make_training_curves()
    make_pareto()
    make_per_class_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
