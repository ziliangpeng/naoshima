import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from sklearn.metrics import accuracy_score, classification_report

import dataloader
import click


# Create the TensorBoard callback
def make_tb(name):
    prefix = name
    # if FLAGS.tag:
    #     prefix += '-' + FLAGS.tag
    log_dir = os.path.join(
        "logs", prefix + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="batch"
    )


# Load dataset
vocab_size = 20000
max_length = 250


X_train, y_train, X_test, y_test = dataloader.load(vocab_size, max_length)

# with tf.device("/GPU:0"):
with tf.device("/CPU:0"):
    # Create the model
    model = Sequential(
        [
            Embedding(vocab_size, 16),
            GlobalAveragePooling1D(),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    """
    layers.Embedding(vocab_size, 128, input_length=max_length),
    and a simple Flatten and Dense layer, will get 86% accuracy
    LSTM/RNN cannot get more than 80% accuracy.
    """
    # Define the RNN model
    model = keras.Sequential(
        [
            layers.Embedding(vocab_size, 128),
            layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
            # layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            # layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    small_model = keras.Sequential(
        [
            layers.Embedding(vocab_size, 128),
            layers.SimpleRNN(128, return_sequences=False), # RNN won't really work.
            # layers.LSTM(128, return_sequences=False), # can go as high as 75% - 80%
            # layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the model
    model = small_model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=42,
        batch_size=256,
        validation_split=0.2,
        verbose=1,
        callbacks=[make_tb("model")],
    )


# # Evaluate the model
@click.command()
@click.option("--evaluate", default=False, help="")
def evaluate():
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# # Plot training and validation accuracy
@click.command()
@click.option("--plot/--no-plot", default=False, help="")
def plot(plot):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.show()


plot()
