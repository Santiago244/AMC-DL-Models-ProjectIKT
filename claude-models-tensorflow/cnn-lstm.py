"""
CNN-LSTM Model for Automatic Modulation Classification (AMC)
Dataset: RadioML 2016.10a / 2018.01a
Input shape: (128, 2) — I/Q samples

Key idea: CNN extracts local spectral/temporal features,
LSTM then captures long-range sequential dependencies in those features.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn_lstm(
    input_shape=(128, 2),
    num_classes=11,
    lstm_units=128,
    dropout_rate=0.5,
    bidirectional=False,
):
    """
    Build a CNN-LSTM hybrid model for AMC.

    Architecture:
        Conv1D × 2 → MaxPool → Conv1D × 2 → MaxPool →
        LSTM (or BiLSTM) → Dense → Output

    Args:
        input_shape   : tuple, (timesteps, channels)
        num_classes   : int, number of modulation classes
        lstm_units    : int, number of LSTM hidden units
        dropout_rate  : float
        bidirectional : bool, wrap LSTM in Bidirectional if True

    Returns:
        keras.Model
    """
    inputs = keras.Input(shape=input_shape, name="iq_input")

    # CNN feature extractor
    # Block 1
    x = layers.Conv1D(
        64, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(1e-4), name="conv1_1"
    )(inputs)
    x = layers.Conv1D(
        64, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(1e-4), name="conv1_2"
    )(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.BatchNormalization(name="bn1")(x)

    # Block 2
    x = layers.Conv1D(
        128, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(1e-4), name="conv2_1"
    )(x)
    x = layers.Conv1D(
        128, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(1e-4), name="conv2_2"
    )(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    # x shape here: (batch, reduced_timesteps, 128) — LSTM input

    # LSTM sequential modelling
    # return_sequences=False → only use the final hidden state
    lstm_layer = layers.LSTM(
        lstm_units,
        return_sequences=False,
        dropout=0.2,               # input dropout inside LSTM
        recurrent_dropout=0.0,     # keep 0 for GPU compatibility (cuDNN)
        name="lstm",
    )
    if bidirectional:
        x = layers.Bidirectional(lstm_layer, name="bilstm")(x)
    else:
        x = lstm_layer(x)

    # Classifier head
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNNLSTM_AMC")
    return model


# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Standard variant
    model = build_cnn_lstm(input_shape=(128, 2), num_classes=11)
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

    # Bidirectional variant
    print("\n--- Bidirectional CNN-LSTM ---")
    model_bi = build_cnn_lstm(
        input_shape=(128, 2), num_classes=11, bidirectional=True
    )
    model_bi.summary()