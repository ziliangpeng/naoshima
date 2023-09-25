import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from sklearn.metrics import accuracy_score, classification_report

from loguru import logger


# hyperparameter tuning
vocab_size = 20000
max_length = 250  # turns out this is the scaling bottleneck
embedding_dim = 128  # This also matters when it gets large. (>512)


def train(e, v, m):
    start_time = datetime.datetime.now()
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=v)

    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=m, padding="post", truncating="post")
    X_test = pad_sequences(X_test, maxlen=m, padding="post", truncating="post")

    with tf.device("/CPU:0"):
        model = keras.Sequential(
            [
                layers.Embedding(vocab_size, e, input_length=m),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        # Compile the model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    # Train the model
    history = model.fit(
        X_train, y_train, epochs=2, batch_size=256, validation_split=0.2, verbose=0
    )
    end_time = datetime.datetime.now()
    diff_time = end_time - start_time
    # print("Hyperparameters/time: ", e, v, m, diff_time)
    # print("=====================================")
    return history.history["val_accuracy"][-1], diff_time


# train(embedding_dim, vocab_size, max_length)


for embedding_dim in [2, 4, 8, 16, 32, 64, 128]:
    for vocab_size in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for max_length in [4, 8, 16, 32, 64, 128, 256]:
            val_acc, train_time = train(embedding_dim, vocab_size, max_length)
            logger.info(
                f"Hyperparameters/time/acc  : {embedding_dim}, {vocab_size}, {max_length}, {train_time}, {val_acc:.4f}"
            )
