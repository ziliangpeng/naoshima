import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import os
import numpy as np
import datetime


# Create the TensorBoard callback
def make_tb(name):
    log_dir = os.path.join(
        "logs", name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="epoch"
    )

# Set random seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Force TensorFlow to use deterministic GPU algorithms
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

datagen.fit(X_train)

def make_resnet_original(input_shape, num_classes, l2_lambda=0.0):
    # Note: l2_lambda=0.0 means no regularization
    regularizer = regularizers.l2(l2_lambda)

    # Define custom ResNet architecture for CIFAR-10
    def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):

        shortcut = x
        if conv_shortcut:
            shortcut = layers.Conv2D(filters, 1, strides=stride, kernel_regularizer=regularizer)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizer)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = resnet_block(x, 64)
    x = layers.Dropout(0.2)(x)
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, stride=2)
    x = layers.Dropout(0.3)(x)
    x = resnet_block(x, 128)
    x = resnet_block(x, 256, stride=2)
    x = layers.Dropout(0.4)(x)
    x = resnet_block(x, 256)
    # trying to add another layer
    # x = resnet_block(x, 512, stride=2)
    # x = resnet_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizer)(x)

    model = models.Model(inputs, x)
    return model

models = {
    'resnet-original': make_resnet_original(input_shape=(32, 32, 3), num_classes=10),
    'resnet-regularization': make_resnet_original(input_shape=(32, 32, 3), num_classes=10, l2_lambda=0.001),
}

# Create and compile the custom ResNet model
MODEL_NAME = 'resnet-regularization'
model = models[MODEL_NAME]
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), callbacks=[make_tb(MODEL_NAME)])
model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=50, validation_data=(X_test, y_test), callbacks=[make_tb(MODEL_NAME)])
