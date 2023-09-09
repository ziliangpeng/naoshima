from tensorflow.keras.datasets import mnist, fashion_mnist
# other ways to load MNIST include using torchvision, or sklearn.
from loguru import logger


def load_mnist(fashion=False):
    # Load the MNIST dataset
    if fashion:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        logger.info("Using fashion dataset")
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        logger.info("Using MNIST dataset")

    image_shape = x_train[0].shape # (28, 28)
    logger.info(f"The shape is: {image_shape}")

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

