"""Test-set evaluation for ERP-ViT 3D Classification experiments.

Loads a saved checkpoint, runs the official test split, and reports:

  - Top-1 overall classification accuracy
  - Per-class accuracy (recall per category)
  - Full sklearn classification report (precision, recall, F1)
  - Confusion matrix saved as a numpy array

Accuracy is the primary comparison metric used in all tables of the
HSDC and SWHDC papers.

CLI usage::

    python -m src.training.evaluate \\
        --config      configs/resnet34_hsdc_mn10.yaml \\
        --checkpoint  experiments/resnet34_hsdc_mn10_seed42/best_checkpoint.pt

References:
    HSDC paper Tables 2–4   — Stringhini et al., IEEE ICIP 2024
    SWHDC paper Tables I–III — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.preprocessing.dataset import build_dataloaders
from src.training.train import build_model, load_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Run *model* on *test_loader* and compute classification metrics.

    The model is evaluated in ``eval()`` mode with ``torch.no_grad()``.

    Args:
        model:        Model with weights already loaded (already on *device*).
        test_loader:  DataLoader for the test split.
        device:       Target device.
        class_names:  Optional list of class name strings (for reporting).
                      If provided, length must equal the number of classes.

    Returns:
        Dict with keys:

        - ``top1_acc``      : float, overall top-1 accuracy (percent)
        - ``per_class_acc`` : dict[int, float], per-class recall (percent)
        - ``all_preds``     : list[int], predicted labels for every sample
        - ``all_targets``   : list[int], ground-truth labels for every sample
        - ``class_names``   : the *class_names* argument (may be ``None``)
        - ``confusion_matrix``: np.ndarray of shape (C, C)

    References:
        HSDC paper §III (Evaluation); SWHDC paper §IV-A
    """
    model.eval()
    all_preds:   list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    preds_arr   = np.array(all_preds)
    targets_arr = np.array(all_targets)

    # Overall top-1 accuracy
    top1_acc = 100.0 * float((preds_arr == targets_arr).mean())

    # Per-class accuracy (recall)
    num_classes = len(class_names) if class_names else int(targets_arr.max()) + 1
    per_class_acc: dict[int, float] = {}
    for cls in range(num_classes):
        mask = targets_arr == cls
        if mask.sum() > 0:
            per_class_acc[cls] = 100.0 * float((preds_arr[mask] == targets_arr[mask]).mean())

    # Confusion matrix
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets_arr.tolist(), preds_arr.tolist()):
        conf_mat[int(t), int(p)] += 1

    return {
        "top1_acc":        top1_acc,
        "per_class_acc":   per_class_acc,
        "all_preds":       all_preds,
        "all_targets":     all_targets,
        "class_names":     class_names,
        "confusion_matrix": conf_mat,
    }


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_evaluation(
    config_path: Path,
    checkpoint_path: Path,
) -> dict:
    """Load a model checkpoint and evaluate it on the official test split.

    Args:
        config_path:      Path to the experiment YAML config file.
        checkpoint_path:  Path to the model checkpoint (``.pt``).

    Returns:
        Evaluation results dict (see :func:`evaluate_model`).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s", device)

    # ------------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------------
    model = build_model(cfg)
    ckpt  = load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    logger.info(
        "Checkpoint: epoch=%d  val_acc=%.2f%%",
        ckpt.get("epoch", -1),
        ckpt.get("val_acc", float("nan")),
    )

    # ------------------------------------------------------------------
    # Build test DataLoader (test split only — never used for model selection)
    # ------------------------------------------------------------------
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]

    loaders = build_dataloaders(
        data_root=Path(data_cfg["data_root"]),
        cache_dir=Path(data_cfg["cache_dir"]),
        pipeline=str(data_cfg["pipeline"]),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        width=int(model_cfg.get("erp_width", 512)),
        height=int(model_cfg.get("erp_height", 256)),
        train_val_split=float(data_cfg.get("train_val_split", 0.8)),
        seed=int(cfg.get("seed", 42)),
    )

    class_names: list[str] = loaders["test"].dataset.classes

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = evaluate_model(model, loaders["test"], device, class_names)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Top-1 accuracy: %.2f%%", results["top1_acc"])
    logger.info("Per-class accuracy:")
    for cls_idx, acc in sorted(results["per_class_acc"].items()):
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        logger.info("  %-24s %.2f%%", name, acc)
    logger.info("=" * 60)

    # sklearn classification report (optional dependency)
    try:
        from sklearn.metrics import classification_report
        report = classification_report(
            results["all_targets"],
            results["all_preds"],
            target_names=class_names,
            zero_division=0,
        )
        logger.info("Classification report:\n%s", report)
    except ImportError:
        logger.warning("scikit-learn not available — skipping classification report.")

    # ------------------------------------------------------------------
    # Save artefacts to the run directory alongside the checkpoint
    # ------------------------------------------------------------------
    run_dir = checkpoint_path.parent

    # Mean class accuracy (macro-averaged recall)
    num_cls = len(class_names)
    per_cls_list = [results["per_class_acc"].get(c, 0.0) for c in range(num_cls)]
    macc = float(np.mean(per_cls_list)) / 100.0   # fraction

    # Parameter count
    try:
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    except Exception:
        params_m = None

    # test_results.json — keys match src/analysis/compare.py expectations
    test_json = {
        "oa":         results["top1_acc"] / 100.0,   # fraction in [0,1]
        "macc":       macc,
        "top1_acc":   results["top1_acc"],            # percent, for backward compat
        "params_m":   params_m,
        "num_classes": num_cls,
    }
    json_path = run_dir / "test_results.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(test_json, jf, indent=2)
    logger.info("Saved test results → %s", json_path)

    # predictions.npz — ground truth and predicted labels for McNemar / confusion matrix
    npz_path = run_dir / "predictions.npz"
    np.savez(
        str(npz_path),
        y_true=np.array(results["all_targets"], dtype=np.int32),
        y_pred=np.array(results["all_preds"],   dtype=np.int32),
    )
    logger.info("Saved predictions → %s", npz_path)

    # confusion_matrix.npy — raw integer counts
    cm_path = run_dir / "confusion_matrix.npy"
    np.save(str(cm_path), results["confusion_matrix"])
    logger.info("Saved confusion matrix → %s", cm_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ERP model on the official test split."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    args = parser.parse_args()
    run_evaluation(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
