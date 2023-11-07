import time
import unittest

from loguru import logger
import dataloader


class TestDataloader(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
