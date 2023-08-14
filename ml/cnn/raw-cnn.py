import numpy as np

import tensorflow as tf

import sys

"""
NOT YET WORKING.
"""

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# It should be numpy format, 60000 images of 28x28 pixels. value 0-256 in each channel.
print("Train data type:", type(x_train))
print("Train data shape:", x_train.shape)
print("Train data size:", x_train.size)
print("Test data type:", type(x_test))
print("Test data shape:", x_test.shape)
print("Test data size:", x_test.size)

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a channel dimension to the images
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# print("Train data type:", type(x_train))
# print("Train data shape:", x_train.shape)

# Convert labels to one-hot encoding
print("yTrain data type:", type(y_train))
print(y_train[0]) # 9
print("yTrain data shape:", y_train.shape)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
print("yTrain data type:", type(y_train))
print(y_train[0]) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
print("yTrain data shape:", y_train.shape)

# sys.exit()

# Define the model architecture
class Conv2D:
    def __init__(self, filters, kernel_size, activation, input_shape=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = input_shape
        self.weights = np.random.randn(filters, kernel_size[0], kernel_size[1], input_shape[-1]) * np.sqrt(2 / np.prod(kernel_size))

    def forward(self, x):
        self.x = x
        self.z = np.zeros((x.shape[0], x.shape[1] - self.kernel_size[0] + 1, x.shape[2] - self.kernel_size[1] + 1, self.filters))
        for i in range(self.filters):
            for j in range(self.x.shape[-1]):
                self.z[..., i] += np.apply_along_axis(lambda m: np.sum(m * self.weights[i, ..., j]), 1, self.x[..., j])
        if self.activation == "relu":
            self.a = np.maximum(self.z, 0)
        else:
            self.a = self.z
        return self.a

    def backward(self, dL_da):
        if self.activation == "relu":
            dL_dz = dL_da * (self.z > 0)
        else:
            dL_dz = dL_da

        dL_dw = np.zeros_like(self.weights)
        for i in range(self.filters):
            for j in range(self.x.shape[-1]):
                for k in range(dL_da.shape[-1]):
                    dL_dw[i, ..., j] += np.sum(self.x[..., j] * dL_dz[..., i, k, np.newaxis], axis=0)

        dL_dx = np.zeros_like(self.x)
        for i in range(self.x.shape[-1]):
            for j in range(self.filters):
                for k in range(dL_da.shape[-1]):
                    dL_dx[..., i] += np.apply_along_axis(lambda m: np.sum(m * self.weights[j, ..., i] * dL_dz[..., j, k]), 1, dL_dz[..., :, k])

        return dL_dx, dL_dw

class MaxPooling2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, x):
        self.x = x
        self.z = np.zeros((x.shape[0], x.shape[1] // self.pool_size[0], x.shape[2] // self.pool_size[1], x.shape[3]))
        for i in range(self.z.shape[1]):
            for j in range(self.z.shape[2]):
                self.z[:, i, j, :] = np.max(self.x[:, i*self.pool_size[0]:(i+1)*self.pool_size[0], j*self.pool_size[1]:(j+1)*self.pool_size[1], :], axis=(1, 2))
        return self.z

class Flatten:
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

class Dense:
    def __init__(self, units, activation, input_shape=None):
        self.units = units
        self.activation = activation
        self.weights = np.random.randn(units, input_shape[-1]) * np.sqrt(2 / input_shape[-1])

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.weights.T)
        if self.activation == "relu":
            self.a = np.maximum(self.z, 0)
        else:
            self.a = self.z
        return self.a

class Softmax:
    def forward(self, x):
        self.x = x
        self.z = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.a = self.z / np.sum(self.z, axis=1, keepdims=True)
        return self.a

# Define the model architecture
model = [
    Conv2D(32, (3, 3), "relu", input_shape=(28, 28, 1)),
    # MaxPooling2D((2, 2)),
    # Conv2D(64, (3, 3), "relu"),
    # MaxPooling2D((2, 2)),
    Flatten(),
    # Dense(128, "relu"),
    Dense(10, "softmax"),
]

# Train the model
batch_size = 32
epochs = 10
learning_rate = 0.01
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        # Forward pass
        x = x_batch
        for layer in model:
            x = layer.forward(x)
        # Compute loss and accuracy
        loss = -np.mean(np.log(x[np.arange(len(x)), y_batch]))
        acc = np.mean(np.argmax(x, axis=1) == y_batch)
        # Backward pass
        grad = np.zeros_like(x)
        grad[np.arange(len(x)), y_batch] = -1 / x[np.arange(len(x)), y_batch]
        for layer in reversed(model):
            grad = layer.backward(grad)
        # Update weights
        for layer in model:
            if hasattr(layer, "weights"):
                layer.weights -= learning_rate * layer.grad_weights
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

# # Evaluate the model on the test set
# x = x_test
# for layer in model:
#     x = layer.forward(x)
# acc = np.mean(np.argmax(x, axis=1) == y_test)
# print(f"Test accuracy = {acc:.4f}")