"""
Implement back prop for this without using TF

model = tf.keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()

# Extract the input features and target labels
X = data.data
y = data.target

# Standardize the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the forward function
def forward(X, W):
    # Calculate the dot product of the input features and the weights
    z = np.dot(X, W)

    # Apply the sigmoid activation function
    output = sigmoid(z)

    return output

# Define the backpropagation function
def backpropagation(X, y, output, W):
    # Calculate the error
    error = y - output

    # Calculate the gradient of the error with respect to the weights
    gradient = error * sigmoid_derivative(output)

    # Update the weights
    W += np.dot(X.T, gradient)

    return W

# Initialize the weights
W = np.random.randn(X.shape[1], 1)

# Train the neural network using backpropagation
num_epochs = 1000
learning_rate = 0.01

for epoch in range(num_epochs):
    # Forward pass
    output = forward(X, W)

    # Backpropagation
    W = backpropagation(X, y, output, W)

    # Compute the loss
    loss = np.mean(np.square(y - output))

    # Print the loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluate the trained model on the training data
y_pred = np.round(forward(X, W))

accuracy = np.mean(y == y_pred)
print(f"Accuracy: {accuracy:.4f}")
