import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a simple DNN model
model = Sequential(
    [
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=SGD(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
)

# Train the model using CPU
with tf.device("/GPU:0"):
    history = model.fit(
        x_train, y_train, epochs=100, batch_size=64, validation_split=0.2
    )

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# print weights
for l in model.layers:
    if not l.get_weights():
        continue
    weights, bias = l.get_weights()
    print("Learned weight:", weights[0][0])
    print("Learned bias:", bias[0])
