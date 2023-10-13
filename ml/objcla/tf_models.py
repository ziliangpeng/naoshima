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
    RandomFlip,
    LeakyReLU,
    concatenate,
    AveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model


def FNN(image_shape):
    return Sequential(
        [
            Flatten(input_shape=image_shape),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


def CNN(image_shape):
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
        Conv2D(
            96,
            3,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
            input_shape=image_shape,
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(
            256,
            3,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(
            384,
            3,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        Conv2D(
            384,
            3,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        Conv2D(
            256,
            3,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        ),
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
            RandomRotation(0.1),
            RandomTranslation(0.1, 0.1),
        ] + layers
    return Sequential(layers)


def VGGNet(image_shape, num_classes, augmentation=False, l2_lambda=0.0):
    """This is actually not working..."""
    model = Sequential()

    if augmentation:
        model.add(RandomRotation(0.1))
        model.add(RandomTranslation(0.1, 0.1))

    # Block 1
    model.add(
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=image_shape)
    )
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # The gen-code is for ImageNet, which is larger and can be pooled more times.
    # This will error out for mnist, since image too small. maybe to intelligent pooling based on image size.
    # # Block 5
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # FC layers
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def inception_module(x, filters):
    # 1x1 Convolution
    conv1 = Conv2D(
        filters=filters[0], kernel_size=(1, 1), padding="same", activation="relu"
    )(x)

    # 1x1 Convolution followed by 3x3 Convolution
    conv3_reduce = Conv2D(
        filters=filters[1], kernel_size=(1, 1), padding="same", activation="relu"
    )(x)
    conv3 = Conv2D(
        filters=filters[2], kernel_size=(3, 3), padding="same", activation="relu"
    )(conv3_reduce)

    # 1x1 Convolution followed by 5x5 Convolution
    conv5_reduce = Conv2D(
        filters=filters[3], kernel_size=(1, 1), padding="same", activation="relu"
    )(x)
    conv5 = Conv2D(
        filters=filters[4], kernel_size=(5, 5), padding="same", activation="relu"
    )(conv5_reduce)

    # 3x3 Max Pooling followed by 1x1 Convolution
    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    pool_proj = Conv2D(
        filters=filters[5], kernel_size=(1, 1), padding="same", activation="relu"
    )(pool)

    # Concatenate all the feature maps and return
    output = concatenate([conv1, conv3, conv5, pool_proj], axis=-1)
    return output


def GoogLeNet(input_shape, num_classes):
    input_layer = Input(input_shape)

    # Initial Conv and MaxPool
    x = Conv2D(64, (7, 7), padding="same", strides=(2, 2), activation="relu")(
        input_layer
    )
    x = MaxPooling2D((3, 3), padding="same", strides=(2, 2))(x)

    # Inception Modules followed by MaxPool
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [128, 128, 192, 96, 64, 64])
    x = MaxPooling2D((3, 3), padding="same", strides=(2, 2))(x)

    # More Inception Modules followed by AveragePool
    x = inception_module(x, [192, 96, 208, 16, 48, 64])
    x = inception_module(x, [160, 112, 224, 24, 64, 64])
    x = inception_module(x, [128, 128, 256, 24, 64, 64])
    x = inception_module(x, [112, 144, 288, 32, 64, 64])
    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = AveragePooling2D((7, 7), padding="same")(x)

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def ResNet(input_shape, num_classes, augmentation=False, l2_lambda=0.0):
    # Note: l2_lambda=0.0 means no regularization
    regularizer = regularizers.l2(l2_lambda)

    # Define custom ResNet architecture for CIFAR-10
    def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
        shortcut = x
        if conv_shortcut:
            # 1x1 convolution to match the dimensions between the residual and identity blocks
            shortcut = layers.Conv2D(
                filters,
                1,
                strides=stride,
                kernel_regularizer=regularizer,
                kernel_initializer=initializers.HeNormal(),
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            padding="same",
            kernel_regularizer=regularizer,
            kernel_initializer=initializers.HeNormal(),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters,
            kernel_size,
            padding="same",
            kernel_regularizer=regularizer,
            kernel_initializer=initializers.HeNormal(),
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = inputs
    if augmentation:
        # Adding a flip and use smaller rotation/translation seems to improve quite a bit.
        x = RandomFlip("horizontal")(x)
        x = RandomRotation(0.05)(x)
        x = RandomTranslation(0.1, 0.1)(x)

    # Note: default initializer is Glorot. But He is better for ReLU.
    x = layers.Conv2D(
        64,
        3,
        padding="same",
        kernel_regularizer=regularizer,
        kernel_initializer=initializers.HeNormal(),
    )(x)
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
    x = layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizer)(
        x
    )

    model = models.Model(inputs, x)
    return model
