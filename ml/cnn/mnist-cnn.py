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


def model1():
    return models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
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


model = model2()

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_images, train_labels, epochs=20, batch_size=128, validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
