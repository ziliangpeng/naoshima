import datetime
import os
import random
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# # Optionally, drop in Fashion MNIST to replace MNIST
(train_images, train_labels), (
    test_images,
    test_labels,
) = datasets.fashion_mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to be suitable for CNN
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Create the TensorBoard callback
def make_tb(name):
    log_dir = os.path.join(
        "logs", name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq='batch'
    )

def alexnet():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def alexnet2():
    # Create AlexNet model
    model = models.Sequential([
        layers.Conv2D(96, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model


def model1():
    return models.Sequential(
        [
            layers.Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(96, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )


def model2():
    return models.Sequential(
        # Fully working NN with ~90% accuracy
        [
            layers.Conv2D(4, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(10, activation="softmax"),
        ]
    )


model = alexnet2()

def run(lr, batch_size, epoch, verbose=False, tensorboard=False):
    # Compile the model
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    cb = []
    if tensorboard:
        cb.append(make_tb(f"mnist-cnn-lr{lr}-bs{batch_size}"))
    history = model.fit(
        train_images, train_labels, epochs=epoch, batch_size=batch_size, validation_split=0.2, verbose=verbose, callbacks=cb
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Lr: {lr}, Batch size: {batch_size}, Test loss: {test_loss}, Test accuracy: {test_accuracy}")


def run_once():
    run(0.001, 128, 1000, verbose=True, tensorboard=True)

def run_many():
    for lr_cnt in range(5):
        for batch_size_cnt in range(5):
            rnd = random.uniform(-2, -5)
            lr = 10 ** rnd
            batch_size = 2 ** (batch_size_cnt + 6)
            run(lr, batch_size, 20)

# run_many()
run_once()