"""
Publication-quality visualisation functions for ERP-ViT 3D Classification.

All functions return ``matplotlib.figure.Figure`` objects and optionally save
to ``experiments/figures/`` as both ``.png`` (300 dpi) and ``.pdf``.

Plot catalogue
--------------
1. ``plot_confusion_matrix``     — row-normalised seaborn heatmap.
2. ``plot_training_curves``      — EMA-smoothed loss/accuracy + LR twin-axis.
3. ``plot_accuracy_comparison``  — grouped bar chart with reference lines.
4. ``plot_pareto``               — parameter-accuracy scatter + Pareto frontier.
5. ``plot_erp_channels``         — 12-channel HSDC ERP grid with channel labels.
6. ``plot_erp_depth``            — single-channel SWHDC depth ERP.
7. ``plot_gradcam``              — Grad-CAM overlay on ERP input.
8. ``plot_ablation``             — bar/line chart for ablation studies.

References:
    Colormaps: matplotlib docs — perceptually uniform colormaps
    Confusion matrix convention: sklearn normalize='true' (row normalisation)
    EMA smoothing: TensorBoard convention, alpha=0.8
    GradCAM for Swin-T: Jacobgilgit/pytorch-grad-cam, swinT_example.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.metrics import ema_smooth, MODELNET10_CLASSES, MODELNET40_CLASSES

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

# Apply once when this module is imported.
# Using a print-safe, colour-blind-friendly palette.
matplotlib.rcParams.update({
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "axes.grid.axis":      "y",
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
})

# Colour palette — accessible, print-safe
_PALETTE = {
    "hsdc":       "#2166ac",   # blue
    "swhdc":      "#d6604d",   # red-orange
    "resnet":     "#4dac26",   # green
    "swin":       "#9970ab",   # purple
    "effnetv2":   "#e08214",   # amber
    "reference":  "#808080",   # grey (literature baselines)
    "proposed":   "#d73027",   # red (your proposed methods)
}

_SAVE_FORMATS = ("png", "pdf")


def _save(fig: plt.Figure, path: Path | None) -> None:
    """Save figure to PNG (300 dpi) and PDF if *path* is given."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in _SAVE_FORMATS:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 1. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a row-normalised confusion matrix as a seaborn heatmap.

    Convention (standard in 3D classification papers):
    - Row normalisation (true label on rows): each row sums to 1.
    - Display as integer percentages (0-100).
    - ``Blues`` sequential colormap: dark = high recall (diagonal).
    - Annotations suppressed for ModelNet40 (40×40 = 1600 cells).

    Args:
        cm:          (C, C) confusion matrix (values in [0,1] if normalised,
                     or raw counts if not).  If raw counts, they are normalised
                     here.
        class_names: Sequence of C class name strings.
        title:       Figure title.
        save_path:   If given, save to this path (extensions added automatically).

    Returns:
        matplotlib Figure.
    """
    C = len(class_names)

    # Ensure row normalisation
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums   # (C, C) fractions

    cm_pct = (cm_norm * 100).astype(int)

    # Layout: MN10 annotated, MN40 annotation suppressed
    annotate = C <= 10
    figsize  = (8, 7) if C <= 10 else (16, 14)
    fontsize = 9 if C <= 10 else 5

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_pct,
        annot=annotate,
        fmt="d",
        cmap="Blues",
        square=True,
        linewidths=0.3 if C <= 10 else 0.1,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=100,
        annot_kws={"size": fontsize},
        ax=ax,
        cbar_kws={"label": "Recall (%)"},
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    df: pd.DataFrame,
    run_name: str = "",
    ema_alpha: float = 0.8,
    best_epoch: int | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot EMA-smoothed loss and accuracy curves with learning rate on a twin axis.

    Layout: two side-by-side panels:
    - Left:  train / val loss  (EMA-smoothed, raw at 30% opacity)
    - Right: train / val accuracy

    Both panels share an optional secondary right y-axis showing the learning
    rate on a log scale (if a ``lr`` column is present in *df*).

    Args:
        df:          Per-epoch metrics DataFrame with columns:
                     ``epoch``, ``train_loss``, ``train_acc``,
                     ``val_loss``, ``val_acc``, (optional) ``lr``.
        run_name:    Used in the figure suptitle.
        ema_alpha:   EMA smoothing factor (TensorBoard default: 0.8).
        best_epoch:  If given, draws a vertical dashed line at this epoch.
        save_path:   Save path (extensions added automatically).

    Returns:
        matplotlib Figure.
    """
    epochs = df["epoch"].values

    # Smooth loss and accuracy curves
    tr_loss_s  = ema_smooth(df["train_loss"].values, ema_alpha)
    val_loss_s = ema_smooth(df["val_loss"].values,   ema_alpha)
    tr_acc_s   = ema_smooth(df["train_acc"].values,  ema_alpha)
    val_acc_s  = ema_smooth(df["val_acc"].values,    ema_alpha)

    has_lr = "lr" in df.columns

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Training curves — {run_name}", fontsize=12, y=1.01)

    # --- Loss panel ---
    ax1 = axes[0]
    ax1.plot(epochs, df["train_loss"].values, color="#2166ac", alpha=0.25, linewidth=0.8)
    ax1.plot(epochs, df["val_loss"].values,   color="#d6604d", alpha=0.25, linewidth=0.8)
    ax1.plot(epochs, tr_loss_s,  color="#2166ac", linewidth=1.8, label="Train loss")
    ax1.plot(epochs, val_loss_s, color="#d6604d", linewidth=1.8, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(loc="upper right")

    if has_lr:
        ax1r = ax1.twinx()
        ax1r.plot(epochs, df["lr"].values, color="#888888", linestyle="--",
                  linewidth=1.0, alpha=0.6, label="LR")
        ax1r.set_yscale("log")
        ax1r.set_ylabel("Learning rate", color="#888888")
        ax1r.tick_params(axis="y", labelcolor="#888888")
        ax1r.spines["right"].set_visible(True)

    if best_epoch is not None:
        ax1.axvline(best_epoch, linestyle=":", color="black", alpha=0.6,
                    label=f"Best (ep {best_epoch})")
        ax1.legend(loc="upper right")

    # --- Accuracy panel ---
    # NOTE: train_acc / val_acc are stored as percentages (0–100) in metrics.csv
    # by train_one_epoch. Do NOT multiply by 100 here.
    ax2 = axes[1]
    ax2.plot(epochs, df["train_acc"].values, color="#2166ac", alpha=0.25, linewidth=0.8)
    ax2.plot(epochs, df["val_acc"].values,   color="#d6604d", alpha=0.25, linewidth=0.8)
    ax2.plot(epochs, tr_acc_s,  color="#2166ac", linewidth=1.8, label="Train acc")
    ax2.plot(epochs, val_acc_s, color="#d6604d", linewidth=1.8, label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")

    if best_epoch is not None:
        ax2.axvline(best_epoch, linestyle=":", color="black", alpha=0.6)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Accuracy comparison bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(
    results: dict[str, dict[str, float]],
    datasets: Sequence[str] = ("MN10", "MN40"),
    metric: str = "oa",
    baselines: dict[str, dict[str, float]] | None = None,
    title: str = "Accuracy Comparison",
    save_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing OA or mAcc across runs and datasets.

    Reference lines are drawn for the HSDC paper baseline (97.1% MN10 / 93.9% MN40)
    and SWHDC paper baseline (94.1% MN10 / 91.9% MN40).

    Args:
        results:  Dict mapping ``run_label → {dataset_name → metric_value}``.
                  Values are fractions in [0,1] or percentages in [0,100].
                  E.g.::
                    {
                      "ResNet-34+HSDC": {"MN10": 0.971, "MN40": 0.939},
                      "Swin-T+HSDC":    {"MN10": 0.952, "MN40": 0.921},
                    }
        datasets: Dataset names to include (default: MN10, MN40).
        metric:   Label for the y-axis (``'oa'`` or ``'macc'``).
        baselines: Optional dict of reference lines in the same format as *results*.
                   Drawn as horizontal dashed lines rather than bars.
        title:    Figure title.
        save_path: Save path.

    Returns:
        matplotlib Figure.
    """
    labels   = list(results.keys())
    n_runs   = len(labels)
    n_ds     = len(datasets)
    x        = np.arange(n_runs)
    bar_w    = 0.35 if n_ds == 2 else 0.6 / n_ds
    offsets  = np.linspace(-(n_ds - 1) * bar_w / 2, (n_ds - 1) * bar_w / 2, n_ds)

    # Normalise values to percentages
    def _pct(v: float) -> float:
        return v * 100 if v <= 1.0 else v

    ds_colors = ["#2166ac", "#d6604d", "#4dac26", "#9970ab"]

    fig, ax = plt.subplots(figsize=(max(8, n_runs * 1.4), 5))

    for di, (ds, offset) in enumerate(zip(datasets, offsets)):
        vals = [_pct(results[run].get(ds, float("nan"))) for run in labels]
        bars = ax.bar(
            x + offset, vals,
            width=bar_w * 0.9,
            color=ds_colors[di % len(ds_colors)],
            label=ds,
            zorder=3,
        )
        # Annotate bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                )

    # Paper baseline reference lines
    _ref_lines = {
        "HSDC baseline MN10": ("MN10", 97.1, "#2166ac", ":"),
        "HSDC baseline MN40": ("MN40", 93.9, "#2166ac", "--"),
        "SWHDC baseline MN10": ("MN10", 94.1, "#d6604d", ":"),
        "SWHDC baseline MN40": ("MN40", 91.9, "#d6604d", "--"),
    }
    plotted_refs: set[str] = set()
    if baselines:
        for bname, bvals in baselines.items():
            for ds in datasets:
                if ds in bvals:
                    v = _pct(bvals[ds])
                    ax.axhline(v, linestyle="--", color="#888888", linewidth=1.0,
                               alpha=0.7, zorder=2)
                    ax.text(n_runs - 0.1, v + 0.15, f"{bname} ({v:.1f}%)",
                            ha="right", va="bottom", fontsize=7, color="#555555")

    metric_label = "Overall Accuracy (%)" if metric == "oa" else "Mean Class Accuracy (%)"
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(title="Dataset")

    # Y-axis: focus on competitive range
    all_vals = [
        _pct(v) for run_vals in results.values() for v in run_vals.values()
        if not np.isnan(v)
    ]
    if all_vals:
        ymin = max(0, min(all_vals) - 3)
        ymax = min(100, max(all_vals) + 3)
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Parameter–Accuracy Pareto plot
# ---------------------------------------------------------------------------

#: Published reference methods for ModelNet40 (params_M, oa_pct, label)
PARETO_REFERENCE_MN40: list[tuple[float, float, str]] = [
    (3.5,  89.2, "PointNet"),
    (1.7,  91.9, "PointNet++"),
    (2.0,  93.8, "CurveNet"),
    (4.5,  93.7, "PointNeXt-S"),
    (12.6, 94.5, "PointMLP"),
    (12.0, 97.6, "View-GCN"),
    (5.3,  93.9, "ResNet-34+HSDC*"),
    (25.5, 91.9, "ResNet-50+SWHDC*"),
    (25.0, 79.7, "PanoFormer ViT"),
]

PARETO_REFERENCE_MN10: list[tuple[float, float, str]] = [
    (3.5,  93.0, "PointNet"),
    (1.7,  94.9, "PointNet++"),
    (5.3,  97.1, "ResNet-34+HSDC*"),
    (25.5, 94.1, "ResNet-50+SWHDC*"),
    (25.0, 85.7, "PanoFormer ViT"),
]


def _pareto_frontier(params: np.ndarray, accs: np.ndarray) -> np.ndarray:
    """Return boolean mask for non-dominated points (Pareto frontier).

    A point (p, a) is Pareto-optimal if no other point has both smaller p
    AND higher a.

    Returns:
        Boolean mask of shape (N,).
    """
    n = len(params)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if params[j] <= params[i] and accs[j] >= accs[i]:
                if params[j] < params[i] or accs[j] > accs[i]:
                    dominated[i] = True
                    break
    return ~dominated


def plot_pareto(
    proposed: dict[str, tuple[float, float]],
    dataset: str = "MN40",
    title: str | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Parameter-accuracy scatter plot with Pareto frontier.

    Args:
        proposed:  Dict mapping ``run_label → (params_M, oa_pct)`` for the
                   proposed methods (plotted as red stars).
        dataset:   ``'MN10'`` or ``'MN40'`` — selects reference method list.
        title:     Figure title (default generated from dataset).
        save_path: Save path.

    Returns:
        matplotlib Figure.
    """
    refs = PARETO_REFERENCE_MN10 if dataset == "MN10" else PARETO_REFERENCE_MN40
    if title is None:
        title = f"Parameter–Accuracy Trade-off on {dataset}"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xscale("log")

    # Reference methods
    ref_params = np.array([r[0] for r in refs])
    ref_accs   = np.array([r[1] for r in refs])
    ref_labels = [r[2] for r in refs]

    # Highlight the paper baselines differently
    baseline_mask = np.array(["*" in lbl for lbl in ref_labels])
    ax.scatter(
        ref_params[~baseline_mask], ref_accs[~baseline_mask],
        c="#888888", s=70, marker="o", zorder=3, label="Literature",
    )
    ax.scatter(
        ref_params[baseline_mask], ref_accs[baseline_mask],
        c="#2166ac", s=90, marker="D", zorder=4, label="Paper baselines",
    )

    for p, a, lbl in zip(ref_params, ref_accs, ref_labels):
        clean_lbl = lbl.replace("*", "")
        ax.annotate(
            clean_lbl, (p, a),
            textcoords="offset points", xytext=(5, 2),
            fontsize=7.5, color="#444444",
        )

    # Proposed methods
    if proposed:
        prop_params = np.array([v[0] for v in proposed.values()])
        prop_accs   = np.array([v[1] for v in proposed.values()])
        ax.scatter(prop_params, prop_accs, c="#d73027", s=140, marker="*",
                   zorder=5, label="Proposed (this work)")
        for lbl, (p, a) in proposed.items():
            ax.annotate(
                lbl, (p, a),
                textcoords="offset points", xytext=(5, 2),
                fontsize=7.5, color="#d73027", fontweight="bold",
            )

    # Pareto frontier across all points
    all_params = np.concatenate([ref_params, np.array([v[0] for v in proposed.values()])]) if proposed else ref_params
    all_accs   = np.concatenate([ref_accs,   np.array([v[1] for v in proposed.values()])]) if proposed else ref_accs
    pareto_mask = _pareto_frontier(all_params, all_accs)
    pareto_pts  = sorted(zip(all_params[pareto_mask], all_accs[pareto_mask]))
    if len(pareto_pts) >= 2:
        pp, pa = zip(*pareto_pts)
        ax.step(pp, pa, where="post", linestyle="--", color="#555555",
                linewidth=1.2, alpha=0.7, label="Pareto frontier")

    ax.set_xlabel("Number of parameters (M)")
    ax.set_ylabel(f"Overall Accuracy (%) on {dataset}")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. ERP 12-channel visualisation (HSDC)
# ---------------------------------------------------------------------------

#: Channel names for the 12-channel HSDC ERP (HSDC paper §II-B)
HSDC_CHANNEL_NAMES: tuple[str, ...] = (
    "d₁ (depth near)",   "Nx₁ (normal x near)", "Ny₁ (normal y near)", "Nz₁ (normal z near)",
    "align₁ (orient.)",  "grad₁ (gradient)",
    "dₙ (depth far)",    "Nxₙ (normal x far)",  "Nyₙ (normal y far)",  "Nzₙ (normal z far)",
    "alignₙ (orient.)",  "gradₙ (gradient)",
)

#: Colormap per channel type
_CHANNEL_CMAPS = (
    "plasma",   # d₁
    "RdBu_r",   # Nx₁
    "RdBu_r",   # Ny₁
    "RdBu_r",   # Nz₁
    "RdBu_r",   # align₁
    "inferno",  # grad₁
    "plasma",   # dₙ
    "RdBu_r",   # Nxₙ
    "RdBu_r",   # Nyₙ
    "RdBu_r",   # Nzₙ
    "RdBu_r",   # alignₙ
    "inferno",  # gradₙ
)


def plot_erp_channels(
    erp: np.ndarray,
    class_name: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Visualise all 12 channels of a HSDC ERP image.

    Layout: 3-row × 4-column grid.  Each subplot uses the channel-appropriate
    colormap (plasma for depth, RdBu_r for signed data, inferno for magnitude).
    Colorbars are shown for every subplot.

    Args:
        erp:        ``(12, H, W)`` or ``(1, H, W)`` float32 numpy array.
        class_name: Shown in the suptitle.
        save_path:  Save path.

    Returns:
        matplotlib Figure.
    """
    C = erp.shape[0]
    if C == 1:
        return plot_erp_depth(erp, class_name=class_name, save_path=save_path)
    if C != 12:
        raise ValueError(f"Expected 12-channel HSDC ERP, got {C} channels.")

    fig, axes = plt.subplots(3, 4, figsize=(18, 8))
    fig.suptitle(
        f"HSDC 12-channel ERP{' — ' + class_name if class_name else ''}",
        fontsize=12, y=1.01,
    )

    for i, ax in enumerate(axes.flat):
        channel = erp[i]   # (H, W)
        cmap    = _CHANNEL_CMAPS[i]
        name    = HSDC_CHANNEL_NAMES[i]

        # Centre diverging colormaps at zero
        if cmap == "RdBu_r":
            vmax = max(abs(channel.min()), abs(channel.max()), 1e-6)
            im = ax.imshow(channel, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        else:
            im = ax.imshow(channel, cmap=cmap, aspect="auto")

        ax.set_title(name, fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Single-channel SWHDC depth ERP
# ---------------------------------------------------------------------------

def plot_erp_depth(
    erp: np.ndarray,
    class_name: str = "",
    add_latitude_grid: bool = True,
    save_path: Path | None = None,
) -> plt.Figure:
    """Visualise a 1-channel SWHDC depth ERP at native 2:1 aspect ratio.

    An optional latitude grid (dashed lines at ±30°, ±60°, equator) is
    overlaid to show where SWHDC latitude weights are highest.

    Args:
        erp:                ``(1, H, W)`` or ``(H, W)`` float32 array.
        class_name:         Shown in the title.
        add_latitude_grid:  Draw latitude reference lines (default True).
        save_path:          Save path.

    Returns:
        matplotlib Figure.
    """
    if erp.ndim == 3:
        erp = erp[0]

    H, W = erp.shape

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(erp, cmap="plasma", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Normalised depth")

    if add_latitude_grid:
        # Latitude lines: φ = π/2 ± lat_deg * π/180 → pixel row = (φ/π) * H - 0.5
        for lat_deg, style in [(0, "-"), (30, "--"), (60, ":")]:
            for sign in ([0] if lat_deg == 0 else [1, -1]):
                phi = np.pi / 2 - sign * np.radians(lat_deg)
                row = phi / np.pi * H - 0.5
                if 0 <= row <= H:
                    label = f"{'+' if sign > 0 else '-' if sign < 0 else ''}{lat_deg}°"
                    ax.axhline(row, linestyle=style, color="white", alpha=0.5, linewidth=0.8)
                    ax.text(W * 0.01, row - 2, label, color="white", fontsize=7, va="bottom")

    ax.set_title(f"SWHDC depth ERP{' — ' + class_name if class_name else ''}")
    ax.set_xlabel("Horizontal angle (ERP columns)")
    ax.set_ylabel("Vertical angle (ERP rows)")
    ax.axis("on")

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Grad-CAM visualisation
# ---------------------------------------------------------------------------

def plot_gradcam(
    grayscale_cam: np.ndarray,
    erp: np.ndarray,
    class_name: str = "",
    pred_class: str = "",
    channel_idx: int = 0,
    alpha: float = 0.45,
    save_path: Path | None = None,
) -> plt.Figure:
    """Overlay a Grad-CAM activation map on an ERP channel.

    The Grad-CAM heatmap (grayscale, in [0, 1]) is blended with the ERP
    image using a jet-like colormap overlay for contrast.

    Args:
        grayscale_cam:  (H, W) Grad-CAM output from ``pytorch-grad-cam``.
        erp:            ``(C, H, W)`` ERP tensor (used as background).
        class_name:     True class label for the title.
        pred_class:     Predicted class label for the title.
        channel_idx:    Which ERP channel to use as background (default 0 = depth).
        alpha:          Opacity of the CAM overlay (default 0.45).
        save_path:      Save path.

    Returns:
        matplotlib Figure.

    Note:
        To generate ``grayscale_cam`` for a Swin-T model::

            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            def reshape_transform(tensor, height=7, width=7):
                result = tensor.reshape(
                    tensor.size(0), height, width, tensor.size(2)
                ).transpose(2, 3).transpose(1, 2)
                return result

            target_layers = [model.backbone.layers[-1].blocks[-1].norm2]
            cam = GradCAM(model=model, target_layers=target_layers,
                          reshape_transform=reshape_transform)
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(class_idx)],
            )[0]
    """
    if erp.ndim == 3:
        bg = erp[channel_idx]
    else:
        bg = erp

    H_cam, W_cam   = grayscale_cam.shape
    H_erp, W_erp   = bg.shape

    # Upsample CAM to ERP resolution if needed
    if (H_cam, W_cam) != (H_erp, W_erp):
        from scipy.ndimage import zoom
        scale_h = H_erp / H_cam
        scale_w = W_erp / W_cam
        grayscale_cam = zoom(grayscale_cam, (scale_h, scale_w), order=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: original ERP background channel
    bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    axes[0].imshow(bg_norm, cmap="plasma", aspect="auto")
    axes[0].set_title(f"Input ERP (ch {channel_idx})")
    axes[0].axis("off")

    # Right: CAM overlay
    axes[1].imshow(bg_norm, cmap="gray", aspect="auto")
    axes[1].imshow(grayscale_cam, cmap="jet", alpha=alpha, aspect="auto",
                   vmin=0, vmax=1)
    correct = class_name == pred_class
    status  = "✓ correct" if correct else f"✗ predicted: {pred_class}"
    axes[1].set_title(f"Grad-CAM — true: {class_name} ({status})")
    axes[1].axis("off")

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Ablation bar/line chart
# ---------------------------------------------------------------------------

def plot_ablation(
    data: dict[str, list[float]],
    x_labels: list[str | int | float],
    x_label: str = "Ablated variable",
    y_label: str = "Overall Accuracy (%)",
    title: str = "Ablation Study",
    baseline: float | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot one or more ablation curves on a shared axes.

    Args:
        data:      Dict mapping ``series_name → [y_values]``.  Each list must
                   have the same length as ``x_labels``.
        x_labels:  X-axis tick labels (e.g. dilation rates, channel counts).
        x_label:   X-axis label.
        y_label:   Y-axis label.
        title:     Figure title.
        baseline:  Optional horizontal reference line (e.g. paper's reported value).
        save_path: Save path.

    Returns:
        matplotlib Figure.
    """
    colors  = list(_PALETTE.values())
    markers = ["o", "s", "D", "^", "v", "<", ">"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(x_labels))

    for idx, (name, values) in enumerate(data.items()):
        pct = [v * 100 if v <= 1.0 else v for v in values]
        ax.plot(
            x, pct,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=1.8,
            markersize=6,
            label=name,
        )
        ax.bar(x, pct, width=0.3,
               color=colors[idx % len(colors)], alpha=0.12, zorder=1)

    if baseline is not None:
        bpct = baseline * 100 if baseline <= 1.0 else baseline
        ax.axhline(bpct, linestyle="--", color="#888888", linewidth=1.2,
                   label=f"Paper baseline ({bpct:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_labels])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if len(data) > 1 or baseline is not None:
        ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 9. Per-class accuracy bar chart (supplementary)
# ---------------------------------------------------------------------------

def plot_per_class_accuracy(
    per_class_acc: np.ndarray,
    class_names: Sequence[str],
    run_label: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of per-class recall, sorted descending.

    Args:
        per_class_acc: (C,) array of per-class recall values in [0, 1].
        class_names:   Sequence of C class name strings.
        run_label:     Used in the title.
        save_path:     Save path.

    Returns:
        matplotlib Figure.
    """
    pct   = per_class_acc * 100
    order = np.argsort(pct)  # ascending → shows worst at top in horizontal bar

    fig, ax = plt.subplots(figsize=(7, max(4, len(class_names) * 0.35)))
    bars = ax.barh(
        np.array(class_names)[order],
        pct[order],
        color="#2166ac",
        alpha=0.8,
    )

    # Annotate values
    for bar, v in zip(bars, pct[order]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8)

    ax.axvline(pct.mean(), linestyle="--", color="#d6604d", linewidth=1.2,
               label=f"Mean ({pct.mean():.1f}%)")
    ax.set_xlabel("Per-class Accuracy (%)")
    ax.set_title(f"Per-class accuracy{' — ' + run_label if run_label else ''}")
    ax.set_xlim(0, 105)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig
