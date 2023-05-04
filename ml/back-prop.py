"""
Implement back prop for this without using TF

model = tf.keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])
"""
import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the neural network architecture
input_shape = X_train.shape[1]
num_classes = 1

# Initialize the weights and biases
weights = [
    np.random.randn(input_shape, num_classes)
]

biases = [
    np.zeros((1, num_classes))
]

learning_rate = 0.1

# Define the forward pass function
def forward(inputs):
    output_activations = sigmoid(np.dot(inputs, weights[0]) + biases[0])
    return output_activations

# Define the backpropagation function
def backpropagation(inputs, output_activations, y):
    # Compute the error between the predicted output and the true output
    output_gradient = (output_activations - y) * sigmoid_derivative(output_activations)

    # Compute the gradients of the weights and biases
    weights_gradient = [
        np.dot(inputs.T, output_gradient)
    ]

    biases_gradient = [
        np.sum(output_gradient, axis=0, keepdims=True)
    ]

    # Update the weights and biases using the gradients
    weights[0] -= learning_rate * weights_gradient[0]
    biases[0] -= learning_rate * biases_gradient[0]

# Train the neural network using backpropagation
for epoch in range(num_epochs):
    # Forward pass
    output_activations = forward(inputs)

    # Backpropagation
    backpropagation(inputs, output_activations, y)

    # Compute the loss
    loss = -np.sum(y * np.log(output_activations) + (1 - y) * np.log(1 - output_activations))

    # Print the loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
