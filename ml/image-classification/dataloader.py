from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from loguru import logger
from functools import partial


def _load(loader, name, onehot):
    (x_train, y_train), (x_test, y_test) = loader()
    logger.info(f"Using {name} dataset")

    image_shape = x_train[0].shape
    logger.info(f"The size is {len(x_train)}")
    logger.info(f"The shape is: {image_shape}")

    if onehot:
        y_train = to_categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = to_categorical(y_test)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


load_cifar10 = partial(_load, cifar10.load_data, "cifar10")
load_cifar100 = partial(_load, cifar100.load_data, "cifar100")
load_mnist = partial(_load, mnist.load_data, "mnist")
load_fashion_mnist = partial(_load, fashion_mnist.load_data, "fashion")


if __name__ == "__main__":
    load_mnist()
    load_cifar10()
