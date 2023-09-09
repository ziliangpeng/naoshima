import click

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import dataloader


@click.command()
@click.option("--fashion", is_flag=True, help="Use fashion dataset instead of MNIST")
def main(fashion):
    x_train, y_train, x_test, y_test = dataloader.load_mnist(fashion=fashion)

    image_shape = x_train[0].shape # (28, 28)

    model = Sequential(
        [
            Flatten(input_shape=image_shape),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )

    with tf.device("/GPU:0"):
        history = model.fit(
            x_train, y_train, epochs=42, batch_size=64, validation_split=0.2
        )

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # # print weights
    # for l in model.layers:
    #     if not l.get_weights():
    #         continue
    #     weights, bias = l.get_weights()
    #     print("Learned weight:", weights[0][0])
    #     print("Learned bias:", bias[0])


if __name__ == "__main__":
    main()
