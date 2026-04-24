"""
utils/visualization.py
Visualization utilities — matches paper Figures 8, 9, 10.
  - Confusion matrix (Fig. 8)
  - Training / validation loss curves (Fig. 9)
  - ROC-AUC curves (Fig. 10)
  - Metric bar charts (Tables 4, 5)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import List, Dict


PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
           "#06B6D4", "#F97316", "#84CC16"]


# ──────────────────────────────────────────────
# 1. Confusion Matrix (Fig. 8)
# ──────────────────────────────────────────────
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────
# 2. Loss / Validation Curves (Fig. 9)
# ──────────────────────────────────────────────
def plot_loss_curves(
    history,              # Keras History object or dict
    title: str = "Training & Validation Loss",
    save_path: str | None = None,
):
    if hasattr(history, "history"):
        history = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["loss"],     label="Train Loss",  color=PALETTE[0])
    axes[0].plot(history["val_loss"], label="Val Loss",    color=PALETTE[1], linestyle="--")
    axes[0].set_title("Loss Curve",   fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    acc_key = "accuracy" if "accuracy" in history else "acc"
    axes[1].plot(history[acc_key],        label="Train Acc",  color=PALETTE[2])
    axes[1].plot(history[f"val_{acc_key}"],label="Val Acc",   color=PALETTE[3], linestyle="--")
    axes[1].set_title("Accuracy Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────
# 3. ROC-AUC Curves (Fig. 10)
# ──────────────────────────────────────────────
def plot_roc_curves(
    models_probs: Dict[str, np.ndarray],   # {"ModelName": prob_array}
    y_true: np.ndarray,
    save_path: str | None = None,
):
    """
    Plot ROC curves for multiple models on the same axes.
    Matches paper Fig. 10 with AUC annotations.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")

    for i, (name, probs) in enumerate(models_probs.items()):
        p = probs[:, 1] if probs.ndim == 2 else probs
        fpr, tpr, _ = roc_curve(y_true, p)
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if "Proposed" in name else 1.5
        ax.plot(fpr, tpr,
                label=f"{name} (AUC = {roc_auc:.3f})",
                color=PALETTE[i % len(PALETTE)], lw=lw)

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────
# 4. Metric comparison bar chart (Tables 4-5)
# ──────────────────────────────────────────────
def plot_metric_comparison(
    results_dict: Dict[str, Dict[str, float]],   # {"Model": {metric: val}}
    metric: str = "accuracy",
    title: str | None = None,
    save_path: str | None = None,
):
    models = list(results_dict.keys())
    values = [results_dict[m].get(metric, 0) for m in models]
    colors = [PALETTE[-1] if "Proposed" in m else PALETTE[0] for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 5))
    bars = ax.bar(models, values, color=colors, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric.title()} Comparison", fontsize=13, fontweight="bold")
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────
# 5. ARWDO convergence curve
# ──────────────────────────────────────────────
def plot_arwdo_convergence(
    history: List[float],
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color=PALETTE[0], lw=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness (Val Loss)", fontsize=12)
    ax.set_title("ARWDO Convergence Curve", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
