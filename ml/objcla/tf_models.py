from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    RandomRotation,
    RandomTranslation,
)


def TfFnn(image_shape):
    return Sequential(
        [
            Flatten(input_shape=image_shape),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


def TfCnn(image_shape):
    return Sequential(
        [
            Conv2D(32, (5, 5), activation="relu", input_shape=image_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(96, (3, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


def AlexNet(image_shape, num_classes, augmentation=False):
    activation = "relu"  # Make if flag
    layers = [
        Conv2D(96, 3, padding="same", activation=activation, input_shape=image_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, 3, padding="same", activation=activation),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(384, 3, padding="same", activation=activation),
        BatchNormalization(),
        Conv2D(384, 3, padding="same", activation=activation),
        BatchNormalization(),
        Conv2D(256, 3, padding="same", activation=activation),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(4096, activation=activation),
        Dropout(0.5),
        Dense(4096, activation=activation),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]

    if augmentation:
        layers = [
            RandomRotation(0.1),
            RandomTranslation(0.1, 0.1),
        ] + layers
    return Sequential(layers)
