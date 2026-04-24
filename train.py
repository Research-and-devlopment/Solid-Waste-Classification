"""
scripts/train.py
End-to-End Training Script
───────────────────────────
Usage:
    python scripts/train.py --config configs/config.yaml

    # Override specific params:
    python scripts/train.py --config configs/config.yaml \\
        --epochs 50 --batch_size 16 --optimizer adam

Flow:
  1. Load config
  2. Build datasets (with augmentation)
  3. [Optional] Run ARWDO hyperparameter tuning
  4. Build hybrid model with (tuned) hyperparameters
  5. Train with early stopping + checkpointing
  6. Evaluate on test set
  7. Save all results, plots, model weights
"""

import os
import sys
import argparse
import logging
import json
import time
import numpy as np
import yaml
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_datasets
from models.hybrid_model import build_hybrid_model
from utils.arwdo_optimizer import tune_hyperparameters
from utils.metrics import compute_metrics, aggregate_runs, print_metrics, print_aggregate
from utils.visualization import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_roc_curves,
    plot_metric_comparison,
    plot_arwdo_convergence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid MSW Classifier")
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--data_dir",    default=None)
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--optimizer",   choices=["arwdo", "adam", "sgd"], default=None)
    p.add_argument("--save_dir",    default="results")
    p.add_argument("--n_runs",      type=int,   default=None)
    p.add_argument("--no_arwdo",    action="store_true",
                   help="Skip ARWDO tuning, use config defaults")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ──────────────────────────────────────────────
# Load config
# ──────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args) -> dict:
    """CLI overrides take precedence over config file."""
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.optimizer:
        cfg["training"]["optimizer"] = args.optimizer
    if args.n_runs:
        cfg["evaluation"]["n_runs"] = args.n_runs
    return cfg


# ──────────────────────────────────────────────
# ARWDO fitness function factory
# ──────────────────────────────────────────────
def make_fitness_fn(train_ds, val_ds, cfg, warmup_epochs=5):
    """
    Returns a function that builds + trains a model for `warmup_epochs`
    and returns the final validation loss.
    This is the fitness function for ARWDO (Eq. 19).
    """
    def fitness_fn(hp: dict) -> float:
        tf.keras.backend.clear_session()
        try:
            model = build_hybrid_model(
                input_shape=tuple(cfg["data"]["image_size"]) + (3,),
                num_classes=cfg["data"]["num_classes"],
                dc_filters=cfg["ae_dc"]["filters"],
                dilation_rates=cfg["ae_dc"]["dilation_rates"],
                gru_units=cfg["residual_gru"]["units"],
                gru_layers=cfg["residual_gru"]["num_layers"],
                gru_alpha=cfg["residual_gru"]["identity_alpha"],
                gru_dropout=hp.get("dropout", cfg["residual_gru"]["dropout"]),
                elm_hidden=int(hp.get("hidden_units", cfg["elm"]["hidden_units"])),
                elm_c=cfg["elm"]["regularization_c"],
                elm_dropout=hp.get("dropout", cfg["training"]["dropout_rate"]),
                learning_rate=hp.get("learning_rate", cfg["training"]["learning_rate"]),
                weight_decay=hp.get("weight_decay", cfg["training"]["weight_decay"]),
            )
            hist = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=warmup_epochs,
                verbose=0,
            )
            return min(hist.history["val_loss"])
        except Exception as e:
            logger.warning(f"[ARWDO fitness] Error: {e}")
            return 1.0  # penalise bad HPs

    return fitness_fn


# ──────────────────────────────────────────────
# Single training run
# ──────────────────────────────────────────────
def train_one_run(cfg, train_ds, val_ds, test_ds, hp: dict, run_idx: int, save_dir: str):
    tf.keras.backend.clear_session()

    model = build_hybrid_model(
        input_shape=tuple(cfg["data"]["image_size"]) + (3,),
        num_classes=cfg["data"]["num_classes"],
        dc_filters=cfg["ae_dc"]["filters"],
        dilation_rates=cfg["ae_dc"]["dilation_rates"],
        gru_units=cfg["residual_gru"]["units"],
        gru_layers=cfg["residual_gru"]["num_layers"],
        gru_alpha=cfg["residual_gru"]["identity_alpha"],
        gru_dropout=hp.get("dropout", cfg["residual_gru"]["dropout"]),
        elm_hidden=int(hp.get("hidden_units", cfg["elm"]["hidden_units"])),
        elm_c=cfg["elm"]["regularization_c"],
        elm_dropout=hp.get("dropout", cfg["training"]["dropout_rate"]),
        learning_rate=hp.get("learning_rate", cfg["training"]["learning_rate"]),
        weight_decay=hp.get("weight_decay", cfg["training"]["weight_decay"]),
    )

    # Callbacks
    ckpt_path = os.path.join(save_dir, "checkpoints", f"run{run_idx}_best.h5")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-7, verbose=0,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, "logs", f"run{run_idx}"),
        ),
    ]

    logger.info(f"\n{'='*55}")
    logger.info(f"  Training Run {run_idx+1} / {cfg['evaluation']['n_runs']}")
    logger.info(f"{'='*55}")

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    logger.info(f"  Training time: {elapsed/3600:.2f} h")

    # ── Evaluate ──────────────────────────────────────────────
    y_true, y_pred, y_prob = [], [], []
    for batch_x, batch_y in test_ds:
        probs = model(batch_x, training=False).numpy()
        y_prob.append(probs)
        y_pred.extend(probs.argmax(axis=1).tolist())
        y_true.extend(batch_y.numpy().argmax(axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.vstack(y_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, prefix=f"Run {run_idx+1} Test Results")

    # ── Save plots ─────────────────────────────────────────────
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_loss_curves(
        history,
        title="Loss / Accuracy Curves",
        save_path=os.path.join(plots_dir, f"run{run_idx}_loss_curves.png"),
    )
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm, cfg["data"]["class_names"],
        title="Confusion Matrix",
        save_path=os.path.join(plots_dir, f"run{run_idx}_confusion_matrix.png"),
    )

    return metrics, history, model, y_true, y_prob


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    cfg   = merge_args(cfg, args)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ── Step 1: Load datasets ─────────────────────────────────
    logger.info("Loading datasets …")
    train_ds, val_ds, test_ds, class_names = get_datasets(
        data_dir=cfg["data"]["data_dir"],
        image_size=tuple(cfg["data"]["image_size"]),
        batch_size=cfg["training"]["batch_size"],
        train_ratio=cfg["data"]["train_split"],
        val_ratio=cfg["data"]["val_split"],
        seed=args.seed,
    )

    # ── Step 2: ARWDO hyperparameter tuning ───────────────────
    if cfg["training"]["optimizer"] == "arwdo" and not args.no_arwdo:
        logger.info("\n🌧  Running ARWDO hyperparameter optimisation …")
        fitness_fn = make_fitness_fn(train_ds, val_ds, cfg)
        best_hp = tune_hyperparameters(
            build_and_eval_fn=fitness_fn,
            population_size=cfg["arwdo"]["population_size"],
            max_iterations=cfg["arwdo"]["max_iterations"],
            seed=args.seed,
        )
    else:
        logger.info("  Skipping ARWDO — using config defaults.")
        best_hp = {
            "learning_rate": cfg["training"]["learning_rate"],
            "hidden_units":  cfg["elm"]["hidden_units"],
            "batch_size":    cfg["training"]["batch_size"],
            "dropout":       cfg["training"]["dropout_rate"],
            "weight_decay":  cfg["training"]["weight_decay"],
        }

    # Save best HPs
    with open(os.path.join(save_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_hp, f, indent=2)

    # ── Step 3: Multi-run training (§4.1, 5 runs) ─────────────
    n_runs = cfg["evaluation"]["n_runs"]
    all_metrics = []
    final_model = None
    all_probs, all_true = None, None

    for run in range(n_runs):
        metrics, history, model, y_true, y_prob = train_one_run(
            cfg, train_ds, val_ds, test_ds, best_hp, run, save_dir
        )
        all_metrics.append(metrics)
        if run == 0:
            final_model = model
            all_probs = y_prob
            all_true  = y_true

    # ── Step 4: Aggregate & report ────────────────────────────
    agg = aggregate_runs(all_metrics)
    logger.info("\n📊 Aggregated Results across %d runs:", n_runs)
    print_aggregate(agg)

    # Save aggregated results
    with open(os.path.join(save_dir, "aggregated_results.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # ── Step 5: ROC curve (Fig. 10) ───────────────────────────
    plot_roc_curves(
        {"Proposed Model": all_probs},
        all_true,
        save_path=os.path.join(save_dir, "plots", "roc_curves.png"),
    )

    # ── Step 6: Save final model ──────────────────────────────
    model_path = os.path.join(save_dir, "final_model.h5")
    if final_model:
        final_model.save(model_path)
        logger.info(f"  Final model saved → {model_path}")

    logger.info("\n✅ Training complete. Results saved to: %s", save_dir)


if __name__ == "__main__":
    main()
