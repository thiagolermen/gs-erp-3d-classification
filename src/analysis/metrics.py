"""
Evaluation metrics and statistical tests for 3D classification experiments.

Implements:
    - Overall Accuracy (OA) and Mean Class Accuracy (mAcc), the two primary
      metrics reported in top-tier 3D classification papers (PointNeXt,
      PointMLP, HSDC paper Table 2, SWHDC paper Table I).
    - Row-normalized confusion matrix (sklearn convention, per-class recall).
    - McNemar's test for comparing two classifiers on a fixed test set
      (exact binomial variant, recommended over chi-squared for small b+c).
    - Exponential Moving Average (EMA) smoothing for training curves.
    - CSV loader for per-epoch experiment logs produced by the training loop.

References:
    PointNeXt — Ma et al., NeurIPS 2022
    McNemar's test — Dietterich 1998 / mlxtend documentation
    HSDC paper Table 2 — Stringhini et al., IEEE ICIP 2024
    SWHDC paper Table I — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    classification_report as sk_classification_report,
)


# ---------------------------------------------------------------------------
# ModelNet class names
# ---------------------------------------------------------------------------

MODELNET10_CLASSES: tuple[str, ...] = (
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
)

MODELNET40_CLASSES: tuple[str, ...] = (
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox",
)


def class_names_for(num_classes: int) -> tuple[str, ...]:
    """Return the canonical class names for ModelNet10 or ModelNet40."""
    if num_classes == 10:
        return MODELNET10_CLASSES
    if num_classes == 40:
        return MODELNET40_CLASSES
    raise ValueError(f"num_classes must be 10 or 40, got {num_classes}")


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Overall Accuracy (OA).

    OA = number of correct predictions / total predictions.
    This is the primary scalar reported in ModelNet10/40 benchmarks.

    Args:
        y_true: (N,) integer ground-truth labels.
        y_pred: (N,) integer predicted labels.

    Returns:
        OA in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_mean_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> float:
    """Compute Mean Class Accuracy (mAcc = macro-averaged recall).

    mAcc = mean over classes of (TP_i / (TP_i + FN_i)).
    mAcc is the more discriminating metric on ModelNet40 because dominant
    classes (desk, dresser, table, night_stand) inflate OA while being
    frequently confused with each other.

    Args:
        y_true:      (N,) ground-truth labels.
        y_pred:      (N,) predicted labels.
        num_classes: If given, labels are assumed to span 0..num_classes-1
                     (ensures absent test classes still contribute 0 recall).

    Returns:
        mAcc in [0, 1].
    """
    labels = np.arange(num_classes) if num_classes else None
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return float(per_class_recall.mean())


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return per-class recall (true positive rate) as a (num_classes,) array.

    Args:
        y_true:      (N,) ground-truth labels.
        y_pred:      (N,) predicted labels.
        num_classes: Total number of classes.

    Returns:
        per_class_recall: (num_classes,) float array in [0, 1].
    """
    cm = sk_confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    return cm.diagonal() / cm.sum(axis=1).clip(min=1)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    normalize: str = "true",
) -> np.ndarray:
    """Compute a (num_classes × num_classes) confusion matrix.

    Row-normalization (``normalize='true'``) is the standard convention in
    3D classification papers: each row sums to 1 and shows the recall per
    true class.

    Args:
        y_true:      (N,) ground-truth labels.
        y_pred:      (N,) predicted labels.
        num_classes: Total number of classes.
        normalize:   ``'true'`` (row, default), ``'pred'`` (column), ``'all'``,
                     or ``None`` (raw counts).

    Returns:
        (num_classes, num_classes) float array.
    """
    return sk_confusion_matrix(
        y_true, y_pred,
        labels=np.arange(num_classes),
        normalize=normalize,
    )


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    """Return a comprehensive classification summary dict.

    Includes OA, mAcc, and a per-class breakdown (precision, recall, F1,
    support) via sklearn's classification_report.

    Args:
        y_true:       (N,) ground-truth labels.
        y_pred:       (N,) predicted labels.
        class_names:  Sequence of class name strings.

    Returns:
        Dict with keys 'oa', 'macc', 'per_class' (DataFrame),
        and 'report' (the raw sklearn string).
    """
    oa    = compute_accuracy(y_true, y_pred)
    macc  = compute_mean_class_accuracy(y_true, y_pred, len(class_names))
    report_dict = sk_classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = sk_classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    per_class_df = pd.DataFrame(
        {cls: report_dict[cls] for cls in class_names}
    ).T[["precision", "recall", "f1-score", "support"]]

    return {
        "oa": oa,
        "macc": macc,
        "per_class": per_class_df,
        "report": report_str,
    }


# ---------------------------------------------------------------------------
# Statistical significance — McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_target: np.ndarray,
    y_model1: np.ndarray,
    y_model2: np.ndarray,
    exact: bool = True,
) -> tuple[float, float]:
    """McNemar's test for comparing two classifiers on the same fixed test set.

    McNemar is the correct choice when two models are evaluated on the same
    fixed test set (e.g., ModelNet10 908 samples or ModelNet40 2468 samples).
    It uses only the discordant pairs where the two models disagree.

    The contingency table:
        b = cases where model1 correct, model2 wrong
        c = cases where model1 wrong,   model2 correct

    If b + c >= 25: chi-squared approximation (exact=False).
    If b + c <  25: exact binomial test (exact=True) — conservative default.

    Args:
        y_target:  (N,) ground-truth labels.
        y_model1:  (N,) predictions from model 1.
        y_model2:  (N,) predictions from model 2.
        exact:     Use exact binomial test (default True; safer for small b+c).

    Returns:
        (test_statistic, p_value). For exact=True, statistic is min(b,c).
        Reject H₀ (models differ) at p < 0.05.
    """
    correct1 = (y_model1 == y_target)
    correct2 = (y_model2 == y_target)

    b = int(np.sum(correct1 & ~correct2))   # model1 right, model2 wrong
    c = int(np.sum(~correct1 & correct2))   # model1 wrong, model2 right

    if b + c == 0:
        return 0.0, 1.0   # identical predictions

    if exact:
        # Exact binomial: H₀ is b ~ Binomial(b+c, 0.5)
        # Two-tailed: p = 2 * min(P(X<=min(b,c)), P(X>=max(b,c)))
        result = stats.binomtest(min(b, c), b + c, p=0.5, alternative="two-sided")
        return float(min(b, c)), float(result.pvalue)
    else:
        # Chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p    = float(1 - stats.chi2.cdf(chi2, df=1))
        return float(chi2), p


def mcnemar_pairwise(
    predictions: dict[str, np.ndarray],
    y_target: np.ndarray,
    exact: bool = True,
) -> pd.DataFrame:
    """Compute pairwise McNemar p-values for all pairs of models.

    Args:
        predictions: Dict mapping run_name → (N,) predicted labels.
        y_target:    (N,) ground-truth labels.
        exact:       Use exact test (see ``mcnemar_test``).

    Returns:
        Symmetric DataFrame of p-values indexed by run_name.
    """
    names = list(predictions.keys())
    n = len(names)
    pvals = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            _, p = mcnemar_test(
                y_target, predictions[names[i]], predictions[names[j]], exact=exact
            )
            pvals[i, j] = pvals[j, i] = p

    return pd.DataFrame(pvals, index=names, columns=names)


# ---------------------------------------------------------------------------
# EMA smoothing for training curves
# ---------------------------------------------------------------------------

def ema_smooth(values: np.ndarray | list[float], alpha: float = 0.8) -> np.ndarray:
    """Exponential Moving Average smoothing for training curves.

    The update rule (identical to TensorBoard's smoothing):
        smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * raw[t]

    Args:
        values: 1-D array of raw metric values (e.g. per-epoch train loss).
        alpha:  Smoothing factor in [0, 1).
                0.6 = moderate (retains visible fluctuations).
                0.8 = standard (TensorBoard default).
                0.9 = aggressive.

    Returns:
        smoothed: 1-D float array of the same length.
    """
    values = np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for t in range(1, len(values)):
        smoothed[t] = alpha * smoothed[t - 1] + (1.0 - alpha) * values[t]
    return smoothed.astype(np.float32)


# ---------------------------------------------------------------------------
# Experiment log loader
# ---------------------------------------------------------------------------

def load_run_metrics(run_dir: Path) -> pd.DataFrame:
    """Load per-epoch training metrics from an experiment run directory.

    Expects ``<run_dir>/metrics.csv`` with at minimum the columns:
        epoch, train_loss, train_acc, val_loss, val_acc, lr

    Args:
        run_dir: Path to the experiment run directory.

    Returns:
        DataFrame with per-epoch metrics, sorted by epoch.

    Raises:
        FileNotFoundError: If ``metrics.csv`` does not exist.
    """
    csv_path = Path(run_dir) / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found in '{run_dir}'")
    df = pd.read_csv(csv_path)
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def load_test_results(run_dir: Path) -> dict[str, float]:
    """Load final test-set results from ``<run_dir>/test_results.json``.

    Args:
        run_dir: Path to the experiment run directory.

    Returns:
        Dict with at least keys ``'oa'`` and ``'macc'``.

    Raises:
        FileNotFoundError: If ``test_results.json`` does not exist.
    """
    import json
    json_path = Path(run_dir) / "test_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"test_results.json not found in '{run_dir}'")
    with open(json_path) as f:
        return json.load(f)


def load_predictions(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ground-truth and predicted labels from ``<run_dir>/predictions.npz``.

    Args:
        run_dir: Path to the experiment run directory.

    Returns:
        (y_true, y_pred): two (N,) integer arrays.

    Raises:
        FileNotFoundError: If ``predictions.npz`` does not exist.
    """
    npz_path = Path(run_dir) / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"predictions.npz not found in '{run_dir}'")
    data = np.load(str(npz_path))
    return data["y_true"], data["y_pred"]
