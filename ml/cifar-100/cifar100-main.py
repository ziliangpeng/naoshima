import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import random
import os
import numpy as np
import datetime
import re

# Checkpoints
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch{epoch:04d}.h5'),
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)


# Create the TensorBoard callback
def make_tb(name):
    log_dir = os.path.join(
        "logs", name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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


# Load and preprocess the CIFAR-100 dataset
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
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


def make_resnet(input_shape, num_classes, l2_lambda=0.0):
    # Note: l2_lambda=0.0 means no regularization
    regularizer = regularizers.l2(l2_lambda)

    # Define custom ResNet architecture for CIFAR-100
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

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x)
    return model

models = {
    'resnet-simple': make_resnet(input_shape=(32, 32, 3), num_classes=100),
    # 'resnet-regularization': make_resnet_original(input_shape=(32, 32, 3), num_classes=10, l2_lambda=0.00005),
    # 'densenet': make_densenet(input_shape=(32, 32, 3), num_classes=10),
}
MODEL_NAME = 'resnet-simple'
# Create and compile the custom ResNet model
model = models[MODEL_NAME]
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if not checkpoint_files:
        return None

    checkpoint_epochs = [int(re.findall(r'epoch(\d+)', f)[0]) for f in checkpoint_files]
    latest_epoch = max(checkpoint_epochs)
    latest_checkpoint = f'model_epoch{latest_epoch:04d}.h5'

    return os.path.join(checkpoint_dir, latest_checkpoint)

latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

# Load the latest checkpoint
# latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
print(latest_checkpoint)
last_epoch = 0
if latest_checkpoint:
    print(f'Restoring weights from {latest_checkpoint}')
    model.load_weights(latest_checkpoint)

    # Get the last completed epoch number from the checkpoint file name
    last_epoch = int(latest_checkpoint.split('epoch')[-1].split('.')[0])

# Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))
model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=1000, initial_epoch=last_epoch, validation_data=(X_test, y_test), callbacks=[checkpoint_callback, make_tb(MODEL_NAME)])
