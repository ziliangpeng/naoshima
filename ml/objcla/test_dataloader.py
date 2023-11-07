import sys
import time
import unittest

from loguru import logger
import dataloader


class TestDataloader(unittest.TestCase):
    def setUp(self):
        # Add a sink to stderr or stdout with level DEBUG
        logger.remove()  # Remove default handlers to prevent duplicate logs
        logger.add(sys.stderr, level="DEBUG")

    def test_categorical_speed(self):
        _, y_train, _, _ = dataloader.load_mnist(onehot=False)

        start_time = time.time()
        dataloader.categorical(y_train)
        end_time = time.time()
        logger.info(f"Time taken for sklearn categorical: {end_time - start_time}")

        start_time = time.time()
        dataloader.categorical_tf(y_train)
        end_time = time.time()
        logger.info(f"Time taken for tf categorical: {end_time - start_time}")

    def test_load_all(self):
        dataloader.load_mnist(onehot=False)
        dataloader.load_fashion_mnist(onehot=False)
        dataloader.load_cifar10(onehot=False)
        dataloader.load_cifar100(onehot=False)


if __name__ == "__main__":
    unittest.main()
