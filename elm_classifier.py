"""
models/elm_classifier.py
Extreme Learning Machine (ELM) Classifier
──────────────────────────────────────────
Paper §3.6:
  - Single-hidden-layer feedforward network
  - Hidden layer weights/biases assigned randomly (not tuned)
  - Output weights computed analytically via Moore–Penrose pseudoinverse (Eq. 14)
  - Regularisation parameter C (Eq. 15)
  - Output: Softmax probabilities (Eq. 17)
  - Loss: Cross-entropy with L2 regularisation (Eq. 18)

Hyperparameters tuned by ARWDO:
  learning_rate, hidden_units, batch_size, dropout, weight_decay
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ──────────────────────────────────────────────
# ELM Layer  (analytical weight computation)
# ──────────────────────────────────────────────
class ELMLayer(layers.Layer):
    """
    Classic ELM hidden layer.
    Input weights (W) and biases (b) are randomly initialised and FROZEN.
    Only output weights (beta) are trained/computed.

    Forward pass:
        H = g(X @ W^T + b)     — hidden layer activation (Eq. 13)
        beta = H* @ O           — Moore-Penrose pseudoinverse (Eq. 14)
        f(y) = H @ beta         — final output (Eq. 16)
    """

    def __init__(self, hidden_units: int, activation: str = "sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation_fn = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Random input weights — NOT trainable (ELM core principle)
        self.W = self.add_weight(
            name="elm_W",
            shape=(input_dim, self.hidden_units),
            initializer="glorot_uniform",
            trainable=False,
        )
        self.b = self.add_weight(
            name="elm_b",
            shape=(self.hidden_units,),
            initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # Eq. 12-13: Hidden layer output matrix H
        H = self.activation_fn(x @ self.W + self.b)
        return H

    def compute_output_weights(self, H: np.ndarray, Y: np.ndarray, C: float = 1.0):
        """
        Batch analytical solution for beta (Eq. 14–15).
            beta = (I/C + H^T @ H)^{-1} @ H^T @ Y
        """
        N = H.shape[0]
        A = np.eye(self.hidden_units) / C + H.T @ H
        beta = np.linalg.pinv(A) @ H.T @ Y      # Eq. 14-15
        return beta


# ──────────────────────────────────────────────
# ELM-based Deep Feedforward Classifier
# ──────────────────────────────────────────────
class ELMClassifier(layers.Layer):
    """
    Deep feedforward classifier built on ELM principles.
    Combines standard trainable Dense layers (for gradient optimisation)
    with an ELM hidden layer for fast approximation.

    Output: (batch, num_classes) — softmax probabilities (Eq. 17)
    """

    def __init__(
        self,
        hidden_units: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        regularisation_c: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.elm_hidden = ELMLayer(hidden_units, activation="sigmoid", name="elm_hidden")
        self.dropout    = layers.Dropout(dropout)
        self.bn         = layers.BatchNormalization()
        self.output_dense = layers.Dense(
            num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(1.0 / (2.0 * regularisation_c)),
            name="elm_output",
        )

    def call(self, x, training=False):
        # ELM hidden layer (fixed weights)
        H = self.elm_hidden(x)
        H = self.bn(H, training=training)
        H = self.dropout(H, training=training)
        # Trainable output layer (gradient-optimised)
        logits = self.output_dense(H)
        return tf.nn.softmax(logits, name="softmax_out")    # Eq. 17


# ──────────────────────────────────────────────
# Standalone ELM (batch analytical, no backprop)
# ──────────────────────────────────────────────
class BatchELM:
    """
    Pure ELM: no gradient descent.
    Use when you want to compute output weights analytically
    on the entire training set in one shot.

    Useful for ablation studies comparing ELM vs back-prop.
    """

    def __init__(self, hidden_units: int = 256, C: float = 1.0,
                 activation: str = "sigmoid"):
        self.hidden_units = hidden_units
        self.C = C
        self.activation = getattr(np, activation) if hasattr(np, activation) \
            else lambda x: 1 / (1 + np.exp(-x))
        self.W = None
        self.b = None
        self.beta = None

    def _hidden(self, X):
        return self.activation(X @ self.W + self.b)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: (N, input_dim)
        Y: (N, num_classes) — one-hot
        """
        n, d = X.shape
        self.W = np.random.randn(d, self.hidden_units) * 0.1
        self.b = np.random.randn(1, self.hidden_units) * 0.1
        H = self._hidden(X)
        # Eq. 15
        A = np.eye(self.hidden_units) / self.C + H.T @ H
        self.beta = np.linalg.solve(A, H.T @ Y)
        return self

    def predict_proba(self, X: np.ndarray):
        H = self._hidden(X)
        logits = H @ self.beta
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray):
        return self.predict_proba(X).argmax(axis=1)
