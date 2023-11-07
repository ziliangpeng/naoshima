from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.datasets import cifar10, cifar100
from loguru import logger
from functools import partial
from datasets import load_dataset

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def categorical_tf(y):
    # to be deleted.
    from tensorflow.keras.utils import to_categorical as tf_categorical

    return tf_categorical(y)


def categorical(y):
    y = y.reshape(-1, 1)
    # Create the encoder
    encoder = OneHotEncoder(
        sparse_output=False
    )  # sparse=False to ensure output is a dense matrix

    # Fit and transform the labels
    y_onehot = encoder.fit_transform(y)

    return y_onehot


def load_mnist_hf(onehot=True):
    dataset = load_dataset("mnist")
    train_data = dataset["train"]
    test_data = dataset["test"]

    x_train = np.array(train_data["image"]) / 255.0
    y_train = np.array(train_data["label"])

    x_test = np.array(test_data["image"]) / 255.0
    y_test = np.array(test_data["label"])

    # replace this to not use keras
    if onehot:
        y_train = categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = categorical(y_test)

    return x_train, y_train, x_test, y_test


def load_cifar10_hf(onehot=True):
    logger.info(f"Using cifar10 dataset from hf")
    dataset = load_dataset("cifar10")
    train_data = dataset["train"]
    test_data = dataset["test"]

    x_train = np.array(train_data["img"]) / 255.0
    y_train = np.array(train_data["label"])

    x_test = np.array(test_data["img"]) / 255.0
    y_test = np.array(test_data["label"])

    if onehot:
        y_train = categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = categorical(y_test)

    return x_train, y_train, x_test, y_test


def _load_keras(loader, name, onehot):
    (x_train, y_train), (x_test, y_test) = loader()
    logger.info(f"Using {name} dataset from keras")

    image_shape = x_train[0].shape
    logger.info(f"The size is {len(x_train)}")
    logger.info(f"The shape is: {image_shape}")

    if onehot:
        y_train = categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = categorical(y_test)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


load_cifar10 = partial(_load_keras, cifar10.load_data, "cifar10")
load_cifar100 = partial(_load_keras, cifar100.load_data, "cifar100")
load_mnist = partial(_load_keras, mnist.load_data, "mnist")
load_fashion_mnist = partial(_load_keras, fashion_mnist.load_data, "fashion")


if __name__ == "__main__":
    # TODO: convert these to unit tests

    load_mnist(onehot=True)
    load_cifar10(onehot=False)
    X_train, y_train, X_test, y_test = load_mnist(onehot=True)
    logger.info(type(X_train[0]))  # type should be numpy.narray
    logger.info(X_train[0].shape)
    x_train_hf, y_train_hf, x_test_hf, y_test_hf = load_mnist_hf()
    assert np.allclose(X_train, x_train_hf)

    # not work, since order seems to be different?
    # X_train, y_train, X_test, y_test = load_cifar10(onehot=True)
    # x_train_hf, y_train_hf, x_test_hf, y_test_hf = load_cifar10_hf()
    # assert np.allclose(X_train, x_train_hf)

    X_train, y_train, X_test, y_test = load_mnist(onehot=False)
    logger.info(y_train.shape)
    logger.info(y_train[0])
    logger.info(categorical(y_train)[0])
    logger.info(categorical_tf(y_train)[0])
    logger.info(type(categorical(y_train)))
    logger.info(type(categorical_tf(y_train)))
    assert np.array_equal(categorical(y_train), categorical_tf(y_train))
