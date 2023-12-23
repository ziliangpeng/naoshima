from loguru import logger
from sklearn.preprocessing import OneHotEncoder

DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

def categorical_tf(y):
    # to be deleted.
    from tensorflow.keras.utils import to_categorical as tf_categorical

    return tf_categorical(y)


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

def _load_keras(loader, name, onehot):
    (x_train, y_train), (x_test, y_test) = loader()
    logger.info(f"Loading {name} dataset from keras")

    image_shape = x_train[0].shape
    logger.info(f"The size is {len(x_train)}")
    logger.info(f"The shape is: {image_shape}")

    if onehot:
        y_train = categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = categorical(y_test)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

for d in DATASETS:
    try:
        from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

        # globals()[f"load_{d}_tf"] = partial(_load_keras, mnist.load_data, d)
    except ImportError:
        logger.warning(f"Cannot import tensorflow for the {d} dataset.")