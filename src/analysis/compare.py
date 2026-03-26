"""
Cross-run comparison utilities and LaTeX table generation.

Functions
---------
load_all_runs          — Scan experiments/ and load all completed run metrics.
build_comparison_table — Combine runs into a Pandas DataFrame for analysis.
generate_latex_table   — Export a publication-ready LaTeX comparison table.
compute_multi_seed_stats — Mean ± std over multiple seeds for the same config.

The LaTeX output matches Table 2 of the HSDC paper (Stringhini et al., 2024):
columns are Method, Input, OA (%), mAcc (%), and Params (M).
Best result is bold (\\textbf{}), second-best underlined (\\underline{}).

All saved tables go under ``experiments/tables/`` as ``.tex`` files.

References:
    HSDC paper Table 2 — Stringhini et al., IEEE ICIP 2024
    SWHDC paper Table I — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.analysis.metrics import (
    load_run_metrics,
    load_test_results,
    load_predictions,
    compute_accuracy,
    compute_mean_class_accuracy,
    mcnemar_pairwise,
)


# ---------------------------------------------------------------------------
# Published benchmark table (hardcoded for reference lines)
# ---------------------------------------------------------------------------

#: Literature baselines — add to comparison tables without experiment files.
LITERATURE_BASELINES: list[dict[str, object]] = [
    # HSDC paper Table 2
    {
        "method":      "HSDCNet (ResNet-34 + HSDC)",
        "input":       "12-ch ERP",
        "mn10_oa":     97.1,
        "mn40_oa":     93.9,
        "mn40_macc":   None,
        "params_m":    5.3,
        "source":      "HSDC paper Table 2",
    },
    # SWHDC paper Table I
    {
        "method":      "ResNet-50 + SWHDC",
        "input":       "1-ch depth ERP",
        "mn10_oa":     94.1,
        "mn40_oa":     91.9,
        "mn40_macc":   None,
        "params_m":    25.5,
        "source":      "SWHDC paper Table I",
    },
    # SWHDC paper Table III
    {
        "method":      "PanoFormer ViT + FC",
        "input":       "1-ch depth ERP",
        "mn10_oa":     85.7,
        "mn40_oa":     79.7,
        "mn40_macc":   None,
        "params_m":    None,
        "source":      "SWHDC paper Table III",
    },
    # HSDC paper Table 2
    {
        "method":      "PointMLP",
        "input":       "Point cloud",
        "mn10_oa":     None,
        "mn40_oa":     94.5,
        "mn40_macc":   91.5,
        "params_m":    12.6,
        "source":      "HSDC paper Table 2",
    },
    # HSDC paper Table 2
    {
        "method":      "View-GCN (20 views)",
        "input":       "Multi-view",
        "mn10_oa":     None,
        "mn40_oa":     97.6,
        "mn40_macc":   None,
        "params_m":    12.0,
        "source":      "HSDC paper Table 2",
    },
]


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def load_all_runs(
    experiments_dir: Path,
    require_test_results: bool = True,
) -> dict[str, dict[str, object]]:
    """Scan *experiments_dir* and load all completed experiment runs.

    A run directory is considered complete if it contains ``metrics.csv``
    and optionally ``test_results.json``.

    Args:
        experiments_dir:      Root experiments directory.
        require_test_results: Only include runs with ``test_results.json``.

    Returns:
        Dict mapping ``run_name → {'metrics_df': DataFrame, 'test': dict,
        'y_true': array, 'y_pred': array}``.
        Missing files result in ``None`` values for those keys.
    """
    experiments_dir = Path(experiments_dir)
    runs: dict[str, dict[str, object]] = {}

    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name in ("figures", "tables"):
            continue

        test_json = run_dir / "test_results.json"
        metrics_csv = run_dir / "metrics.csv"

        if require_test_results and not test_json.exists():
            continue

        entry: dict[str, object] = {"run_dir": run_dir}

        # Per-epoch training metrics
        try:
            entry["metrics_df"] = load_run_metrics(run_dir)
        except FileNotFoundError:
            entry["metrics_df"] = None

        # Final test results
        try:
            entry["test"] = load_test_results(run_dir)
        except FileNotFoundError:
            entry["test"] = None

        # Predictions (for McNemar test)
        try:
            y_true, y_pred = load_predictions(run_dir)
            entry["y_true"] = y_true
            entry["y_pred"] = y_pred
        except FileNotFoundError:
            entry["y_true"] = None
            entry["y_pred"] = None

        runs[run_dir.name] = entry

    return runs


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------

def build_comparison_table(
    runs: dict[str, dict[str, object]],
    dataset: str = "MN40",
    include_literature: bool = True,
) -> pd.DataFrame:
    """Build a comparison DataFrame combining experiment runs and literature.

    Columns: ``method``, ``input``, ``oa``, ``macc``, ``params_m``, ``source``.

    Args:
        runs:                Dict from ``load_all_runs()``.
        dataset:             ``'MN10'`` or ``'MN40'``.
        include_literature:  Prepend literature baseline rows.

    Returns:
        DataFrame sorted by ``oa`` descending.
    """
    rows: list[dict[str, object]] = []

    # Literature baselines
    if include_literature:
        for b in LITERATURE_BASELINES:
            oa_key  = "mn10_oa" if dataset == "MN10" else "mn40_oa"
            rows.append({
                "method":   b["method"],
                "input":    b["input"],
                "oa":       b[oa_key],
                "macc":     b.get("mn40_macc"),
                "params_m": b.get("params_m"),
                "source":   b["source"],
            })

    # Experiment runs
    for run_name, entry in runs.items():
        test = entry.get("test")
        if test is None:
            continue

        # Parse run_name to infer method/input labels
        method, input_desc = _infer_labels(run_name)

        rows.append({
            "method":   method,
            "input":    input_desc,
            "oa":       test.get("oa", test.get("test_oa")),
            "macc":     test.get("macc", test.get("test_macc")),
            "params_m": test.get("params_m"),
            "source":   "this work",
        })

    df = pd.DataFrame(rows)

    # Convert OA/mAcc fractions to percentages if needed
    for col in ("oa", "macc"):
        if col in df.columns:
            mask = df[col].notna() & (df[col] <= 1.0)
            df.loc[mask, col] = df.loc[mask, col] * 100

    df = df.sort_values("oa", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    return df


def _infer_labels(run_name: str) -> tuple[str, str]:
    """Infer a human-readable (method, input) pair from a run directory name."""
    name = run_name.lower()
    # Method label
    if "resnet34" in name or "resnet_34" in name:
        backbone = "ResNet-34"
    elif "resnet50" in name or "resnet_50" in name:
        backbone = "ResNet-50"
    else:
        backbone = run_name

    block = "HSDC" if "hsdc" in name else ("SWHDC" if "swhdc" in name else "")
    method = f"{backbone} + {block}" if block else backbone

    # Input label — both pipelines now use radiance field ERP
    input_desc = "8-shell RF-ERP"

    return method, input_desc


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def compute_multi_seed_stats(
    runs: dict[str, dict[str, object]],
    base_name: str,
) -> dict[str, float | None]:
    """Compute mean ± std for runs sharing a common base name (different seeds).

    Args:
        runs:      Dict from ``load_all_runs()``.
        base_name: Base run name without seed suffix
                   (e.g. ``'resnet34_hsdc_mn10'`` matches
                   ``'resnet34_hsdc_mn10_seed42'``, ``'…_seed0'``, etc.).

    Returns:
        Dict with keys ``'oa_mean'``, ``'oa_std'``, ``'macc_mean'``,
        ``'macc_std'``, ``'n_seeds'``.
    """
    matching = {
        name: entry for name, entry in runs.items()
        if name.startswith(base_name) and entry.get("test") is not None
    }

    if not matching:
        return {"oa_mean": None, "oa_std": None, "macc_mean": None,
                "macc_std": None, "n_seeds": 0}

    oas   = [e["test"].get("oa", e["test"].get("test_oa")) for e in matching.values()]
    maccs = [e["test"].get("macc", e["test"].get("test_macc")) for e in matching.values()]

    oas   = [v * 100 if v is not None and v <= 1.0 else v for v in oas]
    maccs = [v * 100 if v is not None and v <= 1.0 else v for v in maccs]

    oas_arr   = np.array([v for v in oas   if v is not None])
    maccs_arr = np.array([v for v in maccs if v is not None])

    return {
        "oa_mean":   float(oas_arr.mean())   if len(oas_arr)   else None,
        "oa_std":    float(oas_arr.std())    if len(oas_arr) > 1 else None,
        "macc_mean": float(maccs_arr.mean()) if len(maccs_arr) else None,
        "macc_std":  float(maccs_arr.std())  if len(maccs_arr) > 1 else None,
        "n_seeds":   len(matching),
    }


# ---------------------------------------------------------------------------
# McNemar significance matrix
# ---------------------------------------------------------------------------

def build_mcnemar_table(
    runs: dict[str, dict[str, object]],
    y_target: np.ndarray,
) -> pd.DataFrame:
    """Build a pairwise McNemar p-value matrix for all runs with predictions.

    Args:
        runs:     Dict from ``load_all_runs()``.
        y_target: Ground-truth labels for the test set.

    Returns:
        Symmetric DataFrame of p-values (NaN if predictions not available).
    """
    preds = {
        name: entry["y_pred"]
        for name, entry in runs.items()
        if entry.get("y_pred") is not None
    }
    if len(preds) < 2:
        return pd.DataFrame()
    return mcnemar_pairwise(preds, y_target, exact=True)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_table(
    df: pd.DataFrame,
    dataset: str = "MN40",
    caption: str | None = None,
    label: str | None = None,
    save_path: Path | None = None,
) -> str:
    """Generate a publication-quality LaTeX comparison table.

    Matches the format of Table 2 in the HSDC paper (Stringhini et al., 2024):
    - Columns: Method, Input, OA (%), mAcc (%), Params (M)
    - \\textbf{} for best, \\underline{} for second-best result in each column.

    Args:
        df:        Comparison DataFrame from ``build_comparison_table()``.
        dataset:   Used in caption and label.
        caption:   LaTeX caption string.
        label:     LaTeX \\label{} string.
        save_path: If given, save to this path as a ``.tex`` file.

    Returns:
        LaTeX table string.
    """
    if caption is None:
        caption = (
            f"3D object classification results on {dataset}. "
            "OA = Overall Accuracy, mAcc = Mean Class Accuracy. "
            "Best result in \\textbf{{bold}}, second-best \\underline{{underlined}}."
        )
    if label is None:
        label = f"tab:results_{dataset.lower()}"

    # Format helper
    def _fmt(val: float | None, fmt: str = ".1f") -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return f"{val:{fmt}}"

    # Find best and second-best for OA and mAcc
    oa_vals   = df["oa"].dropna().values
    macc_vals = df["macc"].dropna().values

    oa_best    = float(np.sort(oa_vals)[-1])   if len(oa_vals)   >= 1 else None
    oa_second  = float(np.sort(oa_vals)[-2])   if len(oa_vals)   >= 2 else None
    mac_best   = float(np.sort(macc_vals)[-1]) if len(macc_vals) >= 1 else None
    mac_second = float(np.sort(macc_vals)[-2]) if len(macc_vals) >= 2 else None

    def _mark_oa(val: float | None) -> str:
        s = _fmt(val)
        if val is None or np.isnan(val):
            return s
        if oa_best is not None and abs(val - oa_best) < 0.05:
            return f"\\textbf{{{s}}}"
        if oa_second is not None and abs(val - oa_second) < 0.05:
            return f"\\underline{{{s}}}"
        return s

    def _mark_macc(val: float | None) -> str:
        s = _fmt(val)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return s
        if mac_best is not None and abs(val - mac_best) < 0.05:
            return f"\\textbf{{{s}}}"
        if mac_second is not None and abs(val - mac_second) < 0.05:
            return f"\\underline{{{s}}}"
        return s

    # Separate literature (with \\midrule) from "this work"
    lit_rows  = df[df.get("source", pd.Series([""] * len(df))) != "this work"] \
                if "source" in df.columns else df
    our_rows  = df[df.get("source", pd.Series([""] * len(df))) == "this work"] \
                if "source" in df.columns else pd.DataFrame()

    lines: list[str] = [
        "% Auto-generated by src/analysis/compare.py",
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{llccc}",
        "    \\toprule",
        "    \\textbf{Method} & \\textbf{Input} & \\textbf{OA (\\%)} & "
        "\\textbf{mAcc (\\%)} & \\textbf{Params (M)} \\\\",
        "    \\midrule",
    ]

    def _row(row: pd.Series) -> str:
        method   = str(row.get("method", "")).replace("_", "\\_").replace("&", "\\&")
        inp      = str(row.get("input", "")).replace("_", "\\_")
        oa_s     = _mark_oa(row.get("oa"))
        macc_s   = _mark_macc(row.get("macc"))
        params_s = _fmt(row.get("params_m"), ".1f")
        return f"    {method} & {inp} & {oa_s} & {macc_s} & {params_s} \\\\"

    for _, row in lit_rows.iterrows():
        lines.append(_row(row))

    if not our_rows.empty:
        lines.append("    \\midrule")
        for _, row in our_rows.iterrows():
            lines.append(_row(row))

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    tex = "\n".join(lines) + "\n"

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(tex)

    return tex


# ---------------------------------------------------------------------------
# Best checkpoint finder
# ---------------------------------------------------------------------------

def find_best_checkpoint(run_dir: Path) -> Path | None:
    """Return the path to the best-validation-accuracy checkpoint in *run_dir*.

    Looks for ``best_checkpoint.pt`` first; falls back to the most recently
    modified ``.pt`` file.

    Args:
        run_dir: Path to the experiment run directory.

    Returns:
        Path to checkpoint, or None if no checkpoint found.
    """
    run_dir = Path(run_dir)
    best = run_dir / "best_checkpoint.pt"
    if best.exists():
        return best

    pts = list(run_dir.glob("*.pt"))
    if not pts:
        return None
    return max(pts, key=lambda p: p.stat().st_mtime)
