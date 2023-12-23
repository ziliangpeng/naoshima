from loguru import logger
from functools import partial
from datasets import load_dataset

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def categorical(y):
    # TODO: would be good to not need sklearn either.
    y = y.reshape(-1, 1)
    # Create the encoder
    encoder = OneHotEncoder(
        sparse_output=False
    )  # sparse=False to ensure output is a dense matrix

    # Fit and transform the labels
    y_onehot = encoder.fit_transform(y)

    return y_onehot


def _load_hf(name, onehot):
    logger.info(f"Loading {name} dataset from hf")
    dataset = load_dataset(name)
    train_data = dataset["train"]
    test_data = dataset["test"]

    # "image" for mnist and fashion_mnist, "img" for cifar10 and cifar100
    data_name = "img" in train_data.column_names and "img" or "image"
    logger.debug(f"data_name is {data_name}")

    # "label" for mnist, fashion_mnist, and cifar10, "coarse_label" or "fine_label" for cifar100
    label_name = "label" in train_data.column_names and "label" or "coarse_label"
    logger.debug(f"label_name is {label_name}")

    x_train = np.array(train_data[data_name]) / 255.0
    y_train = np.array(train_data[label_name])

    x_test = np.array(test_data[data_name]) / 255.0
    y_test = np.array(test_data[label_name])

    if onehot:
        y_train = categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = categorical(y_test)

    return x_train, y_train, x_test, y_test



DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

for d in DATASETS:
    globals()[f"load_{d}_hf"] = partial(_load_hf, d)
    globals()[f"load_{d}"] = partial(_load_hf, d)


if __name__ == "__main__":
    # TODO: convert these to unit tests

    load_mnist(onehot=True)
    load_cifar10(onehot=False)
    X_train, y_train, X_test, y_test = load_mnist(onehot=True)
    logger.info(type(X_train[0]))  # type should be numpy.narray
    logger.info(X_train[0].shape)
    x_train_hf, y_train_hf, x_test_hf, y_test_hf = load_mnist_hf(onehot=True)
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
