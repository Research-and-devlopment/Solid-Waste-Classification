"""
models/ae_dc_block.py
Attention-Evoked Dilated Convolutional Block (AE-DC)
────────────────────────────────────────────────────
Paper §3.4:
  - Pyramid Pooling Layers (PPL) for fixed-size outputs from DC layers
  - Four Dilated Conv branches with rates [1, 2, 4, 8]
  - Self-attention maps fused via Softmax + Concatenation
  - No up-sampling → lower computational overhead
  - Receptive field: J_k = J_{k-1} + (K-1) * stride (Eq. 2)
  - Fused feature: F = Concat(Softmax(F_i) for i in branches) (Eq. 3-4)
"""

import tensorflow as tf
from tensorflow.keras import layers


# ──────────────────────────────────────────────
# Self-Attention (Eq. 3)
# ──────────────────────────────────────────────
class SelfAttention(layers.Layer):
    """Channel-wise self-attention using softmax normalisation."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.query = layers.Conv2D(filters // 8, 1, padding="same")
        self.key   = layers.Conv2D(filters // 8, 1, padding="same")
        self.value = layers.Conv2D(filters,      1, padding="same")
        self.gamma = self.add_weight(name="gamma", shape=(1,),
                                     initializer="zeros", trainable=True)

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        q = tf.reshape(self.query(x), (B, -1, C // 8))
        k = tf.reshape(self.key(x),   (B, -1, C // 8))
        v = tf.reshape(self.value(x), (B, -1, C))

        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True), axis=-1)  # Eq. 3
        out  = tf.matmul(attn, v)
        out  = tf.reshape(out, (B, H, W, C))
        return self.gamma * out + x


# ──────────────────────────────────────────────
# Single Dilated Convolutional Branch
# ──────────────────────────────────────────────
def _dilated_branch(x, filters: int, dilation_rate: int, name_prefix: str):
    """
    One DCR branch: Conv (3×3, dilation=r) → BN → ReLU → Attention.
    kernel_size=3, channels=64 as stated in paper §3.4.
    """
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        dilation_rate=dilation_rate,
        name=f"{name_prefix}_conv_d{dilation_rate}",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn_d{dilation_rate}")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu_d{dilation_rate}")(x)
    return x


# ──────────────────────────────────────────────
# Pyramid Pooling Layer
# ──────────────────────────────────────────────
class PyramidPooling(layers.Layer):
    """
    Spatial Pyramid Pooling to generate fixed-size outputs.
    Pools at scales [1, 2, 3, 6], upsamples back, concatenates.
    """

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.pool_sizes = [1, 2, 3, 6]
        self.conv_layers = [
            layers.Conv2D(filters // 4, 1, padding="same",
                          name=f"ppl_conv_{s}") for s in self.pool_sizes
        ]
        self.bn_layers = [
            layers.BatchNormalization(name=f"ppl_bn_{s}") for s in self.pool_sizes
        ]

    def call(self, x):
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        pooled = []
        for ps, conv, bn in zip(self.pool_sizes, self.conv_layers, self.bn_layers):
            p = tf.keras.layers.AveragePooling2D(pool_size=ps, strides=ps, padding="same")(x)
            p = conv(p)
            p = bn(p)
            p = tf.image.resize(p, (h, w))
            pooled.append(p)
        return tf.concat([x] + pooled, axis=-1)


# ──────────────────────────────────────────────
# AE-DC Block  (main export)
# ──────────────────────────────────────────────
class AEDCBlock(layers.Layer):
    """
    Attention-Evoked Dilated Convolutional Block.

    Architecture (paper Fig. 6):
        Input
         ├─ Branch d=1  ──→ SelfAttention ──→
         ├─ Branch d=2  ──→ SelfAttention ──→  Concat → PPL → Conv(1×1) → Output
         ├─ Branch d=4  ──→ SelfAttention ──→
         └─ Branch d=8  ──→ SelfAttention ──→

    Receptive field per layer (Eq. 2):
        J_k = J_{k-1} + (K-1) * t  where K=3, t=dilation_rate
    """

    def __init__(
        self,
        filters: int = 64,
        dilation_rates=(1, 2, 4, 8),
        use_pyramid_pooling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters       = filters
        self.dilation_rates = dilation_rates
        self.use_ppl       = use_pyramid_pooling

        self.attention_layers = [
            SelfAttention(filters, name=f"ae_dc_attn_d{d}") for d in dilation_rates
        ]
        self.ppl = PyramidPooling(filters, name="ae_dc_ppl") if use_pyramid_pooling else None

        # Pointwise conv to project concatenated features back to `filters`
        n_concat = filters * len(dilation_rates)
        if use_pyramid_pooling:
            n_concat += filters   # PPL adds extra channels (handled dynamically below)
        self.proj_conv = layers.Conv2D(filters, 1, padding="same", name="ae_dc_proj")
        self.proj_bn   = layers.BatchNormalization(name="ae_dc_proj_bn")
        self.proj_act  = layers.ReLU(name="ae_dc_proj_relu")

    def call(self, x, training=False):
        branch_outputs = []
        for d, attn in zip(self.dilation_rates, self.attention_layers):
            b = _dilated_branch(x, self.filters, d, name_prefix="ae_dc")
            b = attn(b)                          # Softmax attention (Eq. 3)
            branch_outputs.append(b)

        # Concatenate all attention-weighted features (Eq. 4)
        fused = tf.concat(branch_outputs, axis=-1)

        if self.use_ppl:
            fused = self.ppl(fused)

        out = self.proj_conv(fused)
        out = self.proj_bn(out, training=training)
        out = self.proj_act(out)
        return out


# ──────────────────────────────────────────────
# Functional helper for model building
# ──────────────────────────────────────────────
def build_ae_dc_extractor(input_shape=(224, 224, 3), filters=64):
    """
    Wrap AE-DC inside a standard Keras Sequential/functional model.
    Returns a Keras Model: Image → Feature Map.
    """
    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    # Stem: standard conv block before dilated branches
    x = layers.Conv2D(32, 3, strides=2, padding="same", name="stem_conv1")(inputs)
    x = layers.BatchNormalization(name="stem_bn1")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", name="stem_conv2")(x)
    x = layers.BatchNormalization(name="stem_bn2")(x)
    x = layers.ReLU()(x)

    # AE-DC Block (spatial feature extraction)
    x = AEDCBlock(filters=filters, name="ae_dc_block")(x)

    # Global Average Pooling → flatten for GRU input
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    return tf.keras.Model(inputs, x, name="AE_DC_Extractor")
