"""
models/residual_gru.py
Residual Gated Recurrent Units (Res-GRU)
─────────────────────────────────────────
Paper §3.5:
  - Motivated by ResNet skip connections
  - Adds residual path around GRU block to speed convergence
  - Removes vanishing gradient problem
  - Three Conv layers + BN inside residual block
  - Parameterised ReLU (PReLU) activation
  - Modified hidden state (Eq. 9):
      q = (1-z_t)*q_{t-1} + y_t⊙q̃_t + A*z_t*tanh(1-y_t)
    where A ∈ [0, 0.3] is the identity scalar
"""

import tensorflow as tf
from tensorflow.keras import layers


# ──────────────────────────────────────────────
# PReLU wrapper (paper: "Parameterized ReLU")
# ──────────────────────────────────────────────
class PReLULayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return tf.maximum(0.0, x) + self.alpha * tf.minimum(0.0, x)


# ──────────────────────────────────────────────
# Residual Conv Block inside GRU
# ──────────────────────────────────────────────
class ResidualConvBlock(layers.Layer):
    """
    Three Conv1D layers + BN + PReLU as the residual block.
    Operates on the temporal (sequence) dimension.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(units, 3, padding="same")
        self.bn1   = layers.BatchNormalization()
        self.act1  = PReLULayer()

        self.conv2 = layers.Conv1D(units, 3, padding="same")
        self.bn2   = layers.BatchNormalization()
        self.act2  = PReLULayer()

        self.conv3 = layers.Conv1D(units, 1, padding="same")
        self.bn3   = layers.BatchNormalization()
        self.act3  = PReLULayer()

        # Projection shortcut (if dims differ)
        self.proj  = layers.Conv1D(units, 1, padding="same")

    def call(self, x, training=False):
        shortcut = self.proj(x)

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)
        out = self.act3(out + shortcut)   # skip connection
        return out


# ──────────────────────────────────────────────
# Custom Residual GRU Cell (Eq. 5–9)
# ──────────────────────────────────────────────
class ResidualGRUCell(layers.AbstractRNNCell):
    """
    GRU cell with identity residual term A (Eq. 9):
      q_t = (1-z_t)⊙q_{t-1} + y_t⊙q̃_t + A*z_t*tanh(1 - y_t)
    where A ∈ [0.0, 0.3] controls residual influence.
    """

    def __init__(self, units: int, alpha: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.alpha = alpha

        # Update gate z_t (Eq. 5)
        self.W_z = layers.Dense(units, use_bias=True)
        self.U_z = layers.Dense(units, use_bias=False)

        # Reset gate r_t (Eq. 6)
        self.W_r = layers.Dense(units, use_bias=True)
        self.U_r = layers.Dense(units, use_bias=False)

        # Candidate hidden q̃_t (Eq. 8)
        self.W_q = layers.Dense(units, use_bias=True)
        self.U_q = layers.Dense(units, use_bias=False)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, states):
        q_prev = states[0]

        # Eq. 5 – update gate
        z_t = tf.sigmoid(self.W_z(inputs) + self.U_z(q_prev))
        # Eq. 6 – reset gate
        r_t = tf.sigmoid(self.W_r(inputs) + self.U_r(q_prev))
        # Eq. 8 – candidate hidden state
        q_tilde = tf.tanh(self.W_q(inputs) + self.U_q(r_t * q_prev))
        # Eq. 9 – modified output with residual term
        q_t = ((1.0 - z_t) * q_prev
               + z_t * q_tilde
               + self.alpha * z_t * tf.tanh(1.0 - z_t))
        return q_t, [q_t]


# ──────────────────────────────────────────────
# Stacked Residual GRU Layer
# ──────────────────────────────────────────────
class ResidualGRULayer(layers.Layer):
    """
    Stacks N residual GRU cells with Conv residual blocks between them.

    Input shape:  (batch, timesteps, features)
    Output shape: (batch, units)
    """

    def __init__(
        self,
        units: int = 128,
        num_layers: int = 2,
        alpha: float = 0.2,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.res_convs = [
            ResidualConvBlock(units, name=f"res_conv_{i}") for i in range(num_layers)
        ]
        self.gru_rnns = [
            layers.RNN(
                ResidualGRUCell(units, alpha=alpha),
                return_sequences=(i < num_layers - 1),
                name=f"res_gru_{i}",
            )
            for i in range(num_layers)
        ]
        self.dropouts = [layers.Dropout(dropout) for _ in range(num_layers)]
        self.bn_layers = [layers.BatchNormalization() for _ in range(num_layers)]

    def call(self, x, training=False):
        """x: (batch, timesteps, features)"""
        for res_conv, gru, drop, bn in zip(
            self.res_convs, self.gru_rnns, self.dropouts, self.bn_layers
        ):
            residual = x if len(x.shape) == 3 else tf.expand_dims(x, 1)
            x = res_conv(residual, training=training)
            x = gru(x)
            if len(x.shape) == 3:
                x = bn(x, training=training)
            x = drop(x, training=training)
        return x


# ──────────────────────────────────────────────
# Functional helper
# ──────────────────────────────────────────────
def build_res_gru(input_dim: int, units: int = 128, num_layers: int = 2,
                  alpha: float = 0.2, dropout: float = 0.3):
    """
    Build a small Keras model wrapping ResidualGRULayer.
    Input:  (batch, 1, input_dim)  — single-step sequence
    Output: (batch, units)
    """
    inp = tf.keras.Input(shape=(1, input_dim), name="gru_input")
    out = ResidualGRULayer(units=units, num_layers=num_layers,
                           alpha=alpha, dropout=dropout, name="res_gru")(inp)
    return tf.keras.Model(inp, out, name="ResGRU")
