"""
models/hybrid_model.py
Full Hybrid Model Assembly
──────────────────────────
Paper Fig. 1 — End-to-end pipeline:

  Image Input
    → AE-DC Block  (spatial feature extraction, §3.4)
    → Residual GRU (spatio-temporal refinement, §3.5)
    → ELM Classifier (feedforward classification, §3.6)
    → Softmax Output

Hyperparameters are tuned by ARWDO (§3.6.1, Algorithm 1-2).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from models.ae_dc_block import AEDCBlock
from models.residual_gru import ResidualGRULayer
from models.elm_classifier import ELMClassifier


# ──────────────────────────────────────────────
# Hybrid Model Builder
# ──────────────────────────────────────────────
def build_hybrid_model(
    input_shape=(224, 224, 3),
    num_classes: int = 2,
    # AE-DC params
    dc_filters: int = 64,
    dilation_rates=(1, 2, 4, 8),
    # Residual GRU params
    gru_units: int = 128,
    gru_layers: int = 2,
    gru_alpha: float = 0.2,
    gru_dropout: float = 0.3,
    # ELM params
    elm_hidden: int = 256,
    elm_c: float = 1.0,
    elm_dropout: float = 0.3,
    # Misc
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
) -> Model:
    """
    Constructs the full Keras model.

    Returns a compiled tf.keras.Model ready for model.fit().
    """
    inputs = tf.keras.Input(shape=input_shape, name="waste_image")

    # ── Stage 1: Stem convolutions ────────────────────────────────────────
    x = layers.Conv2D(32, 3, strides=2, padding="same", name="stem_conv1")(inputs)
    x = layers.BatchNormalization(name="stem_bn1")(x)
    x = layers.ReLU(name="stem_act1")(x)

    x = layers.Conv2D(64, 3, strides=1, padding="same", name="stem_conv2")(x)
    x = layers.BatchNormalization(name="stem_bn2")(x)
    x = layers.ReLU(name="stem_act2")(x)
    x = layers.MaxPooling2D(2, name="stem_pool")(x)

    # ── Stage 2: AE-DC Block (spatial feature extraction) ────────────────
    x = AEDCBlock(
        filters=dc_filters,
        dilation_rates=dilation_rates,
        use_pyramid_pooling=True,
        name="ae_dc",
    )(x)

    # ── Stage 3: Intermediate pooling → sequence format ───────────────────
    # GlobalAveragePool: (B, H, W, C) → (B, C)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    # Expand for GRU: (B, C) → (B, 1, C)
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=1), name="expand_seq")(x)

    # ── Stage 4: Residual GRU (spatio-temporal refinement) ────────────────
    x = ResidualGRULayer(
        units=gru_units,
        num_layers=gru_layers,
        alpha=gru_alpha,
        dropout=gru_dropout,
        name="res_gru",
    )(x)

    # ── Stage 5: ELM-based Feedforward Classifier ─────────────────────────
    outputs = ELMClassifier(
        hidden_units=elm_hidden,
        num_classes=num_classes,
        dropout=elm_dropout,
        regularisation_c=elm_c,
        name="elm_classifier",
    )(x)

    model = Model(inputs, outputs, name="HybridMSWClassifier")

    # ── Compile ────────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ──────────────────────────────────────────────
# Ablation / Baseline Variants
# ──────────────────────────────────────────────
def build_baseline_cnn(input_shape=(224, 224, 3), num_classes=2,
                       learning_rate=1e-4) -> Model:
    """Plain CNN baseline (Table 4, row 1)."""
    inp = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out, name="BaselineCNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_gru(input_shape=(224, 224, 3), num_classes=2,
                  learning_rate=1e-4) -> Model:
    """CNN+GRU baseline (Table 4, row 4)."""
    inp = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = tf.expand_dims(x, 1)
    x = layers.GRU(128)(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out, name="CNN_GRU")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ──────────────────────────────────────────────
# Quick model summary helper
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model = build_hybrid_model()
    model.summary(expand_nested=True)
