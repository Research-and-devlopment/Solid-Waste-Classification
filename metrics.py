"""
utils/metrics.py
Evaluation Metrics (Table 3, Equations 20–24)
──────────────────────────────────────────────
  Accuracy    = (TP+TN) / (TP+TN+FP+FN)        Eq. 20
  Recall      = TP / (TP+FN)                    Eq. 21
  Specificity = TN / (TN+FP)                    Eq. 22
  Precision   = TP / (TP+FP)                    Eq. 23
  F1-Score    = 2*Precision*Recall /
                (Precision+Recall)               Eq. 24
  AUC         = area under ROC curve
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from typing import Dict


# ──────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray | None = None) -> Dict[str, float]:
    """
    Compute all paper metrics from flat (non-one-hot) arrays.

    Parameters
    ----------
    y_true : (N,) integer class labels
    y_pred : (N,) integer predicted labels
    y_prob : (N,) or (N, C) predicted probabilities (optional, for AUC)

    Returns
    -------
    dict with keys: accuracy, precision, recall, specificity, f1_score, auc
    """
    cm = confusion_matrix(y_true, y_pred)

    # Binary case: rows/cols = [Organic(0), Recyclable(1)]
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # Multi-class: macro-average per class
        TP = np.diag(cm).sum()
        FP = (cm.sum(axis=0) - np.diag(cm)).sum()
        FN = (cm.sum(axis=1) - np.diag(cm)).sum()
        TN = cm.sum() - TP - FP - FN

    accuracy    = (TP + TN) / (TP + TN + FP + FN + 1e-9)       # Eq. 20
    recall      = TP / (TP + FN + 1e-9)                          # Eq. 21
    specificity = TN / (TN + FP + 1e-9)                          # Eq. 22
    precision   = TP / (TP + FP + 1e-9)                          # Eq. 23
    f1          = (2 * precision * recall) / (precision + recall + 1e-9)  # Eq. 24

    results = dict(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1_score=float(f1),
    )

    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1]
                                    if y_prob.shape[1] == 2 else y_prob,
                                    multi_class="ovr")
            else:
                auc = roc_auc_score(y_true, y_prob)
            results["auc"] = float(auc)
        except Exception:
            results["auc"] = float("nan")

    return results


# ──────────────────────────────────────────────
# Experiment-level aggregation (5 runs, §4.1)
# ──────────────────────────────────────────────
def aggregate_runs(run_results: list[Dict]) -> Dict[str, Dict]:
    """
    Given a list of metric dicts (one per run), compute
    mean, std, best, worst, median, variance for each metric.
    """
    keys = run_results[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in run_results])
        agg[k] = dict(
            mean=float(vals.mean()),
            std=float(vals.std()),
            best=float(vals.max()),
            worst=float(vals.min()),
            median=float(np.median(vals)),
            variance=float(vals.var()),
        )
    return agg


def print_metrics(metrics: Dict, prefix: str = ""):
    """Pretty-print a metrics dictionary."""
    header = f"{'─'*45}"
    print(f"\n{header}")
    if prefix:
        print(f"  {prefix}")
        print(header)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")
        else:
            print(f"  {k:<15}: {v}")
    print(header)


def print_aggregate(agg: Dict):
    """Pretty-print aggregated run statistics (Tables 6–7)."""
    print(f"\n{'─'*70}")
    print(f"  {'Metric':<15} {'Best':>8} {'Worst':>8} {'Mean':>8} "
          f"{'SD':>8} {'Var':>12}")
    print(f"{'─'*70}")
    for k, v in agg.items():
        print(f"  {k:<15} {v['best']:>8.4f} {v['worst']:>8.4f} "
              f"{v['mean']:>8.4f} {v['std']:>8.4f} {v['variance']:>12.2e}")
    print(f"{'─'*70}")
