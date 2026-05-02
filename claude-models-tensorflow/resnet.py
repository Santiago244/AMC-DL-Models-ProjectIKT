"""
ResNet (Residual Network) Model for Automatic Modulation Classification (AMC)
Dataset: RadioML 2016.10a / 2018.01a
Input shape: (128, 2) — I/Q samples

Key idea: residual (skip) connections let gradients flow back
through the network without vanishing, enabling deeper training.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def residual_block(x, filters, kernel_size=3, stride=1, name_prefix="res"):
    """
    One residual block:
        Conv → BN → ReLU → Conv → BN → (+skip) → ReLU

    If input channels ≠ filters, a 1x1 conv projects the shortcut.

    Args:
        x           : input tensor
        filters     : number of Conv1D filters
        kernel_size : convolution kernel size
        stride      : stride for first conv (use 2 to downsample)
        name_prefix : string prefix for layer names

    Returns:
        output tensor
    """
    shortcut = x

    # Main path
    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name_prefix}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)

    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name_prefix}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)

    # Project shortcut if shapes differ (stride or channel change)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{name_prefix}_proj",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    # Merge
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)
    return x


def build_resnet(input_shape=(128, 2), num_classes=11, dropout_rate=0.5):
    """
    Build a ResNet-style 1D model for AMC.

    Architecture:
        Stem Conv → ResBlock×2 (64) → ResBlock×2 (128) →
        ResBlock×2 (256) → GlobalAvgPool → Dense → Output

    Args:
        input_shape  : tuple, (timesteps, channels)
        num_classes  : int, number of modulation classes
        dropout_rate : float

    Returns:
        keras.Model
    """
    inputs = keras.Input(shape=input_shape, name="iq_input")

    # Stem
    x = layers.Conv1D(
        64, kernel_size=7, strides=2, padding="same",
        use_bias=False, name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)

    # Stage 1 — 64 filters
    x = residual_block(x, 64,  name_prefix="stage1_blk1")
    x = residual_block(x, 64,  name_prefix="stage1_blk2")

    # Stage 2 — 128 filters, stride 2 to downsample
    x = residual_block(x, 128, stride=2, name_prefix="stage2_blk1")
    x = residual_block(x, 128, name_prefix="stage2_blk2")

    # Stage 3 — 256 filters, stride 2
    x = residual_block(x, 256, stride=2, name_prefix="stage3_blk1")
    x = residual_block(x, 256, name_prefix="stage3_blk2")

    # Global pooling — collapses the time dimension
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet_AMC")
    return model


# ──────────────────────────────────────────────
if __name__ == "__main__":
    model = build_resnet(input_shape=(128, 2), num_classes=11)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    dummy_x = np.random.randn(8, 128, 2).astype(np.float32)
    dummy_y = tf.keras.utils.to_categorical(
        np.random.randint(0, 11, size=(8,)), num_classes=11
    )
    loss, acc = model.evaluate(dummy_x, dummy_y, verbose=0)
    print(f"Dummy eval — loss: {loss:.4f}, acc: {acc:.4f}")