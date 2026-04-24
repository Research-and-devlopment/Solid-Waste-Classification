"""
scripts/evaluate.py
Evaluate a saved model checkpoint on the test set.
Reproduces Tables 4, 5, 9 and Figures 8, 10 from the paper.

Usage:
    python scripts/evaluate.py \\
        --checkpoint results/final_model.h5 \\
        --test_dir   data/raw/TEST \\
        --config     configs/config.yaml
"""

import os
import sys
import argparse
import json
import numpy as np
import yaml
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_datasets
from models.hybrid_model import (
    build_hybrid_model,
    build_baseline_cnn,
    build_cnn_gru,
)
from utils.metrics import compute_metrics, print_metrics
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_metric_comparison,
)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--test_dir",   default=None)
    p.add_argument("--output_dir", default="results/evaluation")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--run_baselines", action="store_true",
                   help="Also evaluate baseline CNN models for comparison tables")
    return p.parse_args()


# ──────────────────────────────────────────────
# Evaluate one model on test_ds
# ──────────────────────────────────────────────
def evaluate_model(model, test_ds, class_names):
    y_true, y_pred, y_prob = [], [], []
    for batch_x, batch_y in test_ds:
        probs = model(batch_x, training=False).numpy()
        y_prob.append(probs)
        y_pred.extend(probs.argmax(axis=1))
        y_true.extend(batch_y.numpy().argmax(axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.vstack(y_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    return metrics, cm, report, y_true, y_prob


# ──────────────────────────────────────────────
# Statistical significance test (Table 9)
# ──────────────────────────────────────────────
def significance_test(proposed_scores: list, baseline_scores: list,
                       metric: str, baseline_name: str):
    """Paired t-test + Wilcoxon signed-rank test."""
    t_stat, p_t = stats.ttest_rel(proposed_scores, baseline_scores)
    _, p_w = stats.wilcoxon(proposed_scores, baseline_scores)

    print(f"\n  Statistical Test: Proposed vs {baseline_name} ({metric})")
    print(f"    Proposed mean  : {np.mean(proposed_scores):.4f} ± {np.std(proposed_scores):.4f}")
    print(f"    Baseline mean  : {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
    print(f"    t-statistic    : {t_stat:.3f}  |  p (paired t): {p_t:.4f}")
    print(f"    Wilcoxon p     : {p_w:.4f}")
    sig = "✔ Significant" if p_t < 0.05 else "✘ Not Significant"
    print(f"    Result         : {sig}")
    return p_t, p_w


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.test_dir:
        cfg["data"]["data_dir"] = args.test_dir

    # ── Load test data ────────────────────────────────────────
    _, _, test_ds, class_names = get_datasets(
        data_dir=cfg["data"]["data_dir"],
        image_size=tuple(cfg["data"]["image_size"]),
        batch_size=args.batch_size,
    )

    # ── Load proposed model ───────────────────────────────────
    print(f"\n📦 Loading model: {args.checkpoint}")
    proposed_model = tf.keras.models.load_model(args.checkpoint, compile=False)

    metrics, cm, report, y_true, y_prob = evaluate_model(
        proposed_model, test_ds, class_names
    )

    print("\n" + "="*55)
    print("  Proposed Model — Test Set Results")
    print("="*55)
    print_metrics(metrics, "Proposed Hybrid Model")
    print("\nClassification Report:\n", report)

    # Save confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(args.output_dir, "confusion_matrix_proposed.png"),
    )

    all_probs = {"Proposed Model": y_prob}
    results_table = {"Proposed Model": metrics}

    # ── Baseline comparisons (optional) ───────────────────────
    if args.run_baselines:
        print("\n🔬 Evaluating baseline models …")
        baselines = {
            "Baseline CNN": build_baseline_cnn(
                input_shape=tuple(cfg["data"]["image_size"]) + (3,),
                num_classes=cfg["data"]["num_classes"],
            ),
            "CNN + GRU": build_cnn_gru(
                input_shape=tuple(cfg["data"]["image_size"]) + (3,),
                num_classes=cfg["data"]["num_classes"],
            ),
        }
        for name, base_model in baselines.items():
            bm, _, _, _, bp = evaluate_model(base_model, test_ds, class_names)
            print_metrics(bm, name)
            results_table[name] = bm
            all_probs[name] = bp

    # ── ROC curves (Fig. 10) ──────────────────────────────────
    plot_roc_curves(
        all_probs, y_true,
        save_path=os.path.join(args.output_dir, "roc_curves.png"),
    )

    # ── Accuracy bar chart (Table 4/5) ────────────────────────
    plot_metric_comparison(
        results_table, metric="accuracy",
        save_path=os.path.join(args.output_dir, "accuracy_comparison.png"),
    )
    plot_metric_comparison(
        results_table, metric="f1_score",
        save_path=os.path.join(args.output_dir, "f1_comparison.png"),
    )

    # ── Save results JSON ─────────────────────────────────────
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results_table, f, indent=2)

    print(f"\n✅ Evaluation complete. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
