"""
CNN1D Model for Automatic Modulation Classification (AMC)
Dataset: RadioML 2016.10a / 2018.01a
Input shape: (128, 2) — I/Q samples
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn1d(input_shape=(128, 2), num_classes=11, dropout_rate=0.5):
    """
    Build a 1D CNN model for AMC.

    Architecture:
        Conv1D → Conv1D → MaxPool → Conv1D → Conv1D → MaxPool → Flatten → Dense → Output

    Args:
        input_shape  : tuple, (timesteps, channels) — default (128, 2) for I/Q
        num_classes  : int, number of modulation classes
        dropout_rate : float, dropout probability after dense layer

    Returns:
        keras.Model
    """
    inputs = keras.Input(shape=input_shape, name="iq_input")

    # Block 1
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv1_1",
    )(inputs)
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv1_2",
    )(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.BatchNormalization(name="bn1")(x)

    # Block 2
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv2_1",
    )(x)
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv2_2",
    )(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    # Classifier head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN1D_AMC")
    return model

if __name__ == "__main__":
    model = build_cnn1d(input_shape=(128, 2), num_classes=11)
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Dummy forward pass
    dummy_x = np.random.randn(8, 128, 2).astype(np.float32)
    dummy_y = tf.keras.utils.to_categorical(
        np.random.randint(0, 11, size=(8,)), num_classes=11
    )
    loss, acc = model.evaluate(dummy_x, dummy_y, verbose=0)
    print(f"Dummy eval — loss: {loss:.4f}, acc: {acc:.4f}")