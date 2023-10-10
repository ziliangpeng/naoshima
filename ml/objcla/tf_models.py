from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    RandomRotation,
    RandomTranslation,
    LeakyReLU,
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


def AlexNet(image_shape, num_classes, augmentation=False, l2_lambda=0.0):
    # based on viz, relu results in many dead filters.
    # tanh may suffer vanishing gradient.
    # leaky relu seems to learning really well and fast
    # activation = "tanh"  # Make if flag
    activation = LeakyReLU(alpha=0.01)
    regularizer = regularizers.l2(l2_lambda)
    
    layers = [
        Conv2D(96, 3, padding="same", activation=activation, kernel_regularizer=regularizer, input_shape=image_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, 3, padding="same", activation=activation, kernel_regularizer=regularizer),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(384, 3, padding="same", activation=activation, kernel_regularizer=regularizer),
        BatchNormalization(),
        Conv2D(384, 3, padding="same", activation=activation, kernel_regularizer=regularizer),
        BatchNormalization(),
        Conv2D(256, 3, padding="same", activation=activation, kernel_regularizer=regularizer),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(4096, activation=activation, kernel_regularizer=regularizer),
        Dropout(0.5),
        Dense(4096, activation=activation, kernel_regularizer=regularizer),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]

    if augmentation:
        layers = [
            RandomRotation(0.2),
            RandomTranslation(0.2, 0.2),
        ] + layers
    return Sequential(layers)

def ResNet(input_shape, num_classes, augmentation=False, l2_lambda=0.0):
    # Note: l2_lambda=0.0 means no regularization
    regularizer = regularizers.l2(l2_lambda)

    # Define custom ResNet architecture for CIFAR-10
    def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):

        shortcut = x
        if conv_shortcut:
            # 1x1 convolution to match the dimensions between the residual and identity blocks
            shortcut = layers.Conv2D(filters, 1, strides=stride, kernel_regularizer=regularizer, kernel_initializer=initializers.HeNormal())(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=regularizer, kernel_initializer=initializers.HeNormal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizer, kernel_initializer=initializers.HeNormal())(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = inputs
    if augmentation:
        x = RandomRotation(0.2)(x)
        x = RandomTranslation(0.2, 0.2)(x)
        
    # Note: default initializer is Glorot. But He is better for ReLU.
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizer, kernel_initializer=initializers.HeNormal())(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # image sizes are [32, 16, 8]
    # filter counts are [16, 32, 64] # 32 * 16 = 16 * 32 = 8 * 64 = 512
    # good for cifar10.
    x = resnet_block(x, 16)
    for i in range(3):
        x = resnet_block(x, 16, conv_shortcut=False)
    x = layers.Dropout(0.2)(x)
    
    x = resnet_block(x, 32, stride=2)
    for i in range(4):
        x = resnet_block(x, 32, conv_shortcut=False)
    x = layers.Dropout(0.2)(x)

    x = resnet_block(x, 64, stride=2)
    for i in range(6):
        x = resnet_block(x, 64, conv_shortcut=False)
    x = layers.Dropout(0.2)(x)
    
    # trying to add another layer
    x = resnet_block(x, 128, stride=2)
    for i in range(3):
        x = resnet_block(x, 128, conv_shortcut=False)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizer)(x)

    model = models.Model(inputs, x)
    return model