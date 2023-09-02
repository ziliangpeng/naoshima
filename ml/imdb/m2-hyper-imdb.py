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

    model = keras.Sequential(
        [
            layers.Embedding(vocab_size, e, input_length=m),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=2,
        batch_size=256,
        validation_split=0.2,
        verbose=1,
        callbacks=[make_tb("model")],
    )
    end_time = datetime.datetime.now()
    diff_time = end_time - start_time
    print("Hyperparameters/time: ", e, v, m, diff_time)
    print("=====================================")


train(embedding_dim, vocab_size, max_length)


# for embedding_dim in [128, 512, 2048]:
#     for vocab_size in [800, 3200, 12800]:
#         for max_length in [200, 400, 800, 1600, 3200]:
#             train(embedding_dim, vocab_size, max_length)
