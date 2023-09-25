import click

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import dataloader

from tf_models import *


@click.command()
@click.option("--dataset", default="mnist", help="Use fashion dataset instead of MNIST")
def main(dataset):
    loader = getattr(dataloader, f"load_{dataset}")
    x_train, y_train, x_test, y_test = loader(onehot=True)

    if x_train[0].ndim == 2:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    image_shape = x_train[0].shape  # (28, 28)

    model = AlexNet(image_shape, num_classes=y_train.shape[1])

    model.compile(
        optimizer="adam", loss=CategoricalCrossentropy(), metrics=["accuracy"]
    )

    with tf.device("/GPU:0"):
        history = model.fit(
            x_train, y_train, epochs=42, batch_size=64, validation_split=0.2
        )

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
