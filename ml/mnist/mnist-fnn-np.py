import numpy as np
import tensorflow as tf

"""
This is a simple FNN implemented in NumPy.
Setting learning rate too high will cause NaN loss.
Setting 0.000005 can achieve 70+% accuracy in 100 epochs.
"""

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images
train_images = train_images.reshape((60000, 28 * 28)) / 255.0
test_images = test_images.reshape((10000, 28 * 28)) / 255.0

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Define the architecture
input_size = 784
hidden_size = 128
output_size = 10

# Initialize the weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Carefully initialize the weights to avoid extremely large or small values at the start of training.
# A common practice is to use small random values scaled by the square root of the number of units in the preceding layer
W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)

# Hyperparameters
# 0.000005 rate gets NaN in epoch 2 and 3, and improves after epoch 4; very strange.
# 0.000003 has 4 NaN epochs.
# better initial weights can helps.
learning_rate = 0.000005
epochs = 42

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_layer = relu(np.dot(train_images, W1) + b1)
    output_layer = softmax(np.dot(hidden_layer, W2) + b2)

    # Compute the loss
    loss = -np.sum(train_labels * np.log(output_layer)) / train_images.shape[0]

    # Backpropagation
    # My understanding is:
    # backprop calculates all gradient for every place: DLoss / Dsomething.
    # Each gradient of the "something" should be calculatable/differentiable as to how its change affects the loss.
    d_output = output_layer - train_labels
    d_W2 = np.dot(hidden_layer.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)
    d_hidden = np.dot(d_output, W2.T)
    d_hidden[hidden_layer <= 0] = 0  # ReLU derivative
    d_W1 = np.dot(train_images.T, d_hidden)
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

    # Update the weights and biases
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

# Evaluate on the test data
hidden_layer = relu(np.dot(test_images, W1) + b1)
output_layer = softmax(np.dot(hidden_layer, W2) + b2)
predictions = np.argmax(output_layer, axis=1)
true_labels = np.argmax(test_labels, axis=1)
accuracy = np.mean(predictions == true_labels)
print(f"Test Accuracy: {accuracy * 100}%")
