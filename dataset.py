"""
data/dataset.py
Dataset loading, preprocessing, and augmentation pipeline.

Kaggle Solid Waste Dataset:
  - 25,077 images (Organic: 13,966 | Recyclable: 11,111)
  - P-mode images are excluded to avoid colour-banding on resize
  - Augmented set → ~345,870 images
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ──────────────────────────────────────────────
# 1. Exclude P-mode images (paper §3.1)
# ──────────────────────────────────────────────
def _is_valid_image(path: str) -> bool:
    """Return True only for RGB images; reject palette-mode (P) images."""
    try:
        with Image.open(path) as img:
            return img.mode == "RGB"
    except Exception:
        return False


# ──────────────────────────────────────────────
# 2. Build tf.data pipeline
# ──────────────────────────────────────────────
def load_image(path: str, label: int, image_size=(224, 224)):
    """Read, resize, and normalise a single image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.one_hot(label, depth=2)
    return img, label


def build_tf_dataset(paths, labels, image_size, batch_size, augment=False, shuffle=True):
    """Return a batched tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: load_image(p, l, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        ds = ds.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ──────────────────────────────────────────────
# 3. Augmentation (paper §3.2)
# ──────────────────────────────────────────────
def _augment_fn(image, label):
    """
    Geometric + colour-space augmentation applied only to training images.
    Matches the paper: rotation, flip, scale, brightness adjustment.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # Random rotation (±30°) via tf.keras layers
    image = tf.keras.layers.RandomRotation(factor=30 / 360)(
        tf.expand_dims(image, 0)
    )[0]
    image = tf.keras.layers.RandomZoom(height_factor=0.2)(
        tf.expand_dims(image, 0)
    )[0]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ──────────────────────────────────────────────
# 4. High-level dataset builder
# ──────────────────────────────────────────────
def get_datasets(
    data_dir: str,
    image_size=(224, 224),
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
):
    """
    Scan data_dir for TRAIN/ and TEST/ sub-folders (Kaggle structure).
    Returns (train_ds, val_ds, test_ds, class_names).

    Expected layout:
        data_dir/
          TRAIN/O/*.jpg
          TRAIN/R/*.jpg
          TEST/O/*.jpg
          TEST/R/*.jpg
    """
    class_names = ["O", "R"]   # Organic=0, Recyclable=1
    label_map = {c: i for i, c in enumerate(class_names)}

    paths, labels = [], []

    for split in ["TRAIN", "TEST"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in class_names:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if _is_valid_image(fpath):
                    paths.append(fpath)
                    labels.append(label_map[cls])

    paths = np.array(paths)
    labels = np.array(labels)

    # Stratified split: 70 / 10 / 20
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed,
    )
    val_frac = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1 - val_frac),
        stratify=y_tmp,
        random_state=seed,
    )

    print(f"[Dataset] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"[Dataset] Class distribution (train) → "
          f"Organic: {(y_train == 0).sum()}  Recyclable: {(y_train == 1).sum()}")

    train_ds = build_tf_dataset(X_train, y_train, image_size, batch_size, augment=True)
    val_ds   = build_tf_dataset(X_val,   y_val,   image_size, batch_size, augment=False, shuffle=False)
    test_ds  = build_tf_dataset(X_test,  y_test,  image_size, batch_size, augment=False, shuffle=False)

    return train_ds, val_ds, test_ds, ["Organic", "Recyclable"]


# ──────────────────────────────────────────────
# 5. Keras ImageDataGenerator variant (legacy)
# ──────────────────────────────────────────────
def get_generators(data_dir: str, image_size=(224, 224), batch_size: int = 32):
    """
    Alternative: use Keras ImageDataGenerator for flow_from_directory.
    Useful when data is already split into folder-per-class structure.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.125,   # 10% of 80% train = ~10% overall
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "TRAIN"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )
    val_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "TRAIN"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, "TEST"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen
