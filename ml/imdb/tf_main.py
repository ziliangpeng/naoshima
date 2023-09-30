import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Flatten
from sklearn.metrics import accuracy_score, classification_report

import dataloader
from dataloader import VOCAB_SIZE, MAX_LENGTH
import click
from loguru import logger


# Create the TensorBoard callback
def make_tb(name):
    prefix = name
    log_dir = os.path.join(
        "logs", prefix + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="batch"
    )


@click.command()
@click.option("--model", default="mlp", help="")
def train(model):
    X_train, y_train, X_test, y_test = dataloader.load()

    # with tf.device("/GPU:0"):
    with tf.device("/CPU:0"):
        mlp = Sequential(
            [
                # CAn achieve 88% accuracy
                Embedding(VOCAB_SIZE, 1, input_length=MAX_LENGTH),
                # GlobalAveragePooling1D(),
                Flatten(),
                # Dense(4096, activation="tanh"),
                # Dense(512, activation="tanh"),
                # Dense(128, activation="tanh"),
                Dense(32, activation="tanh"),
                Dense(1, activation="sigmoid"),
            ]
        )
        """
        layers.Embedding(vocab_size, 128, input_length=max_length),
        and a simple Flatten and Dense layer, will get 86% accuracy
        LSTM/RNN cannot get more than 80% accuracy.
        """
        lstm = keras.Sequential(
            [
                layers.Embedding(VOCAB_SIZE, 128),
                # layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
                layers.LSTM(128, return_sequences=False),  # can go as high as 75% - 80%
                # layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
                # layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        simplernn = keras.Sequential(
            [
                layers.Embedding(VOCAB_SIZE, 128),
                layers.SimpleRNN(128, return_sequences=False),  # RNN won't really work.
                # layers.LSTM(128, return_sequences=False), # can go as high as 75% - 80%
                # layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        models = {
            "mlp": mlp,
            "lstm": lstm,
            "simplernn": simplernn,
        }

        logger.info(f"Training model: {model}")
        model = models[model]
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

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


if __name__ == "__main__":
    train()
    evaluate()
    plot()
