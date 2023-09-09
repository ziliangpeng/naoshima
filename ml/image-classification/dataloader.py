from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from loguru import logger


def load_cifar10(onehot=False):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    logger.info("Using cifar10 dataset")

    image_shape = x_train[0].shape  # (32, 32)
    logger.info(f"The size is {len(x_train)}")
    logger.info(f"The shape is: {image_shape}")

    if onehot:
        y_train = to_categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = to_categorical(y_test)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


def load_mnist(fashion=False, onehot=False):
    # Load the MNIST dataset
    if fashion:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        logger.info("Using fashion dataset")
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        logger.info("Using MNIST dataset")

    image_shape = x_train[0].shape  # (28, 28)
    logger.info(f"The size is {len(x_train)}")
    logger.info(f"The shape is: {image_shape}")

    if onehot:
        y_train = to_categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = to_categorical(y_test)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    load_mnist()
    load_cifar10()
