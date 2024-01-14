from loguru import logger
from functools import partial
from datasets import load_dataset
import numpy as np
import PIL # requires Pillow to convert PngImageFile to ndarray

def categorical(y):
    l, n = y.shape[0], max(y) + 1
    a = np.repeat(y, n).reshape(l, n)
    b = np.tile(np.arange(n), l).reshape(l, n)
    return (a == b).astype(int)

def _load_hf(name, onehot):
    logger.info(f"Loading {name} dataset from hf")
    dataset = load_dataset(name)
    train_data, test_data = dataset["train"], dataset["test"]

    # "image" for mnist and fashion_mnist, "img" for cifar10 and cifar100
    data_name = "img" in train_data.column_names and "img" or "image"
    logger.debug(f"data_name is {data_name}")
    # "label" for mnist, fashion_mnist, and cifar10, "coarse_label" or "fine_label" for cifar100
    label_name = "label" in train_data.column_names and "label" or "coarse_label"
    logger.debug(f"label_name is {label_name}")

    x_train, x_test = np.array(train_data[data_name]) / 255.0, np.array(test_data[data_name]) / 255.0
    y_train, y_test = np.array(train_data[label_name]), np.array(test_data[label_name])

    if onehot: y_train, y_test = categorical(y_train), categorical(y_test)  # Convert the labels to one-hot encoding

    return x_train, y_train, x_test, y_test

DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

for d in DATASETS:
    globals()[f"load_{d}"] = partial(_load_hf, d)
