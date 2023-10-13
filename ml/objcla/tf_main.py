import datetime
import os
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

from loguru import logger


def make_tb(name):
    prefix = name
    log_dir = os.path.join(
        "logs", prefix + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="batch"
    )


def train(dataset, epoch, model_name):
    logger.info("Using model: " + model_name)
    loader = getattr(dataloader, f"load_{dataset}")
    x_train, y_train, x_test, y_test = loader(onehot=True)

    if x_train[0].ndim == 2:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    image_shape = x_train[0].shape  # (28, 28)

    # ResNet without augmentation, 42 epochs, 83% accuracy
    # ResNet with augmentation, 42 epochs, 79% accuracy. Still climbing.
    resnet = ResNet(image_shape, num_classes=y_train.shape[1], augmentation=True)
    alexnet = AlexNet(image_shape, num_classes=y_train.shape[1], augmentation=True)

    model = {"alexnet": alexnet, "resnet": resnet}[model_name]

    model.compile(
        optimizer="adam", loss=CategoricalCrossentropy(), metrics=["accuracy"]
    )

    with tf.device("/GPU:0"):
        history = model.fit(
            x_train,
            y_train,
            epochs=epoch,
            batch_size=64,
            # validation_split=0.2,
            validation_data=(x_test, y_test),
            callbacks=[make_tb("resnet-" + dataset + "-augmented")],
        )

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    return model


@click.command()
@click.option("--dataset", default="mnist", help="Use fashion dataset instead of MNIST")
@click.option("--epoch", default=42, help="")
@click.option("--model", default="alexnet", help="")
def main(dataset, epoch, model):
    train(dataset, epoch, model)


if __name__ == "__main__":
    main()
