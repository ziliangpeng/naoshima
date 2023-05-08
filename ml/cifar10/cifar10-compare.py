import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import os
import numpy as np
import datetime
import gflags
import sys

gflags.DEFINE_string('tag', '', 'extra tag to add to log file')

gflags.FLAGS(sys.argv)
FLAGS = gflags.FLAGS


# Create the TensorBoard callback
def make_tb(name):
    prefix = name
    if FLAGS.tag:
        prefix += '-' + FLAGS.tag
    log_dir = os.path.join(
        "logs", prefix + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq='batch'
    )

# Set random seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Force TensorFlow to use deterministic GPU algorithms
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def make_densenet(input_shape, num_classes):
    def dense_block(x, growth_rate, num_layers):
        for _ in range(num_layers):
            bn1 = layers.BatchNormalization()(x)
            relu1 = layers.ReLU()(bn1)
            conv1 = layers.Conv2D(4 * growth_rate, (1, 1), padding='same')(relu1)
            bn2 = layers.BatchNormalization()(conv1)
            relu2 = layers.ReLU()(bn2)
            conv2 = layers.Conv2D(growth_rate, (3, 3), padding='same')(relu2)
            x = layers.Concatenate()([x, conv2])
        return x

    def transition_layer(x, compression_factor):
        # reduced_filters = int(tf.shape(x)[-1] * compression_factor)
        reduced_filters = int(x.shape[-1] * compression_factor)
        bn = layers.BatchNormalization()(x)
        relu = layers.ReLU()(bn)
        conv = layers.Conv2D(reduced_filters, (1, 1), padding='same')(relu)
        avg_pool = layers.AveragePooling2D((2, 2), strides=2)(conv)
        return avg_pool

    def DenseNet(input_shape, num_classes, growth_rate=32, num_dense_blocks=3, layers_per_block=4, compression_factor=0.5):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(2 * growth_rate, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        for i in range(num_dense_blocks - 1):
            x = dense_block(x, growth_rate, layers_per_block)
            x = transition_layer(x, compression_factor)

        x = dense_block(x, growth_rate, layers_per_block)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, x)
        return model

    # Create a DenseNet model for CIFAR-10
    model = DenseNet(input_shape=input_shape, num_classes=num_classes)
    return model


def make_resnet_original(input_shape, num_classes, l2_lambda=0.0):
    # Note: l2_lambda=0.0 means no regularization
    regularizer = regularizers.l2(l2_lambda)

    # Define custom ResNet architecture for CIFAR-10
    def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):

        shortcut = x
        if conv_shortcut:
            shortcut = layers.Conv2D(filters, 1, strides=stride, kernel_regularizer=regularizer, kernel_initializer=tf.keras.initializers.HeNormal())(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=regularizer, kernel_initializer=tf.keras.initializers.HeNormal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizer, kernel_initializer=tf.keras.initializers.HeNormal())(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)
    # Note: default initializer is Glorot. But He is better for ReLU.
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizer, kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
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

def main():
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


    models = {
        # This can achive 90+% after ~60 epochs.
        'resnet-simple': make_resnet_original(input_shape=(32, 32, 3), num_classes=10),
        'resnet-regularization': make_resnet_original(input_shape=(32, 32, 3), num_classes=10, l2_lambda=0.00002),
        'densenet': make_densenet(input_shape=(32, 32, 3), num_classes=10),
    }

    # Create and compile the custom ResNet model
    MODEL_NAME = 'resnet-regularization'
    MODEL_NAME = 'densenet'
    MODEL_NAME = 'resnet-simple'
    model = models[MODEL_NAME]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Train the model
    # model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), callbacks=[make_tb(MODEL_NAME)])
    model.fit(datagen.flow(X_train, y_train, batch_size=128), epochs=1000, validation_data=(X_test, y_test), callbacks=[make_tb(MODEL_NAME)])

if __name__ == '__main__':
    main()
