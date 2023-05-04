"""
Implement back prop for this without using TF

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(7, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Load the iris dataset
iris = load_iris()

# Extract the input features and target labels
X = iris.data
y = iris.target

# One-hot encode the target labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Assign the inputs variable
inputs = X


# Define the ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(int)

# Define the softmax function and its derivative
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(output, y):
    return output - y

# Define the neural network architecture
input_shape = (4,)
num_classes = 3
hidden_layer_size = 10
hidden_layer_size2 = 7

# Initialize the weights and biases
weights = [
    np.random.randn(*input_shape, hidden_layer_size),
    np.random.randn(hidden_layer_size, hidden_layer_size2),
    np.random.randn(hidden_layer_size2, num_classes)
]

biases = [
    np.zeros((1, hidden_layer_size)),
    np.zeros((1, hidden_layer_size2)),
    np.zeros((1, num_classes))
]

learning_rate = 0.01

# Define the forward pass function
def forward(inputs):
    hidden_layer_activations = relu(np.dot(inputs, weights[0]) + biases[0])
    hidden_layer_activations2 = relu(np.dot(hidden_layer_activations, weights[1]) + biases[1])
    output_activations = softmax(np.dot(hidden_layer_activations2, weights[2]) + biases[2])
    return hidden_layer_activations, hidden_layer_activations2, output_activations

# Define the backpropagation function
def backpropagation(inputs, hidden_layer_activations, hidden_layer_activations2, output_activations, y):
    # Compute the error between the predicted output and the true output
    output_gradient = softmax_derivative(output_activations, y)

    # Compute the gradient of the second hidden layer with respect to the error
    hidden_gradient2 = np.dot(output_gradient, weights[2].T) * relu_derivative(hidden_layer_activations2)

    # Compute the gradient of the first hidden layer with respect to the error
    hidden_gradient = np.dot(hidden_gradient2, weights[1].T) * relu_derivative(hidden_layer_activations)

    # Compute the gradients of the weights and biases
    weights_gradient = [
        np.dot(inputs.T, hidden_gradient),
        np.dot(hidden_layer_activations.T, hidden_gradient2),
        np.dot(hidden_layer_activations2.T, output_gradient)
    ]

    biases_gradient = [
        np.sum(hidden_gradient, axis=0, keepdims=True),
        np.sum(hidden_gradient2, axis=0, keepdims=True),
        np.sum(output_gradient, axis=0, keepdims=True)
    ]

    # Update the weights and biases using the gradients
    weights[0] -= learning_rate * weights_gradient[0]
    weights[1] -= learning_rate * weights_gradient[1]
    weights[2] -= learning_rate * weights_gradient[2]
    biases[0] -= learning_rate * biases_gradient[0]
    biases[1] -= learning_rate * biases_gradient[1]
    biases[2] -= learning_rate * biases_gradient[2]


# Train the neural network using backpropagation
num_epochs = 10000
epsilon = 1e-8

for epoch in range(num_epochs):
    # Forward pass
    hidden_layer_activations, hidden_layer_activations2, output_activations = forward(inputs)
    # print(hidden_layer_activations, hidden_layer_activations2, output_activations)

    # Backpropagation
    backpropagation(inputs, hidden_layer_activations, hidden_layer_activations2, output_activations, y)

    # Compute the loss
    # loss = -np.sum(y * np.log(output_activations)) # devide by zero
    loss = -np.sum(y * np.log(output_activations + epsilon) + (1 - y) * np.log(1 - output_activations + epsilon))


    # Print the loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
