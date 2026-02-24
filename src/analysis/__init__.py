"""
Result analysis, visualisation, and comparison utilities.

Modules
-------
metrics    — Accuracy helpers, confusion matrix, McNemar test, EMA smoothing.
visualize  — All publication-quality plot generation functions.
compare    — Cross-run comparison, LaTeX table generation.
"""

from src.analysis.metrics import (
    compute_accuracy,
    compute_mean_class_accuracy,
    compute_per_class_accuracy,
    compute_confusion_matrix,
    mcnemar_test,
    ema_smooth,
    load_run_metrics,
)
from src.analysis.compare import (
    load_all_runs,
    build_comparison_table,
    generate_latex_table,
)

__all__ = [
    "compute_accuracy",
    "compute_mean_class_accuracy",
    "compute_per_class_accuracy",
    "compute_confusion_matrix",
    "mcnemar_test",
    "ema_smooth",
    "load_run_metrics",
    "load_all_runs",
    "build_comparison_table",
    "generate_latex_table",
]
