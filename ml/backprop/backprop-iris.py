import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Preprocess the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Preprocess the output labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Define the loss function (categorical crossentropy)
def loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Initialize the weights and biases
weights = [
    np.random.randn(4, 10),
    np.random.randn(10, 7),
    np.random.randn(7, 3),
]

biases = [
    np.zeros(10),
    np.zeros(7),
    np.zeros(3),
]

# Training settings
learning_rate = 0.001
num_epochs = 1000

# Gradient descent optimization (backpropagation)
for epoch in range(num_epochs):
    # Forward pass
    a1 = np.dot(X_train, weights[0]) + biases[0]
    z1 = relu(a1)
    a2 = np.dot(z1, weights[1]) + biases[1]
    z2 = relu(a2)
    a3 = np.dot(z2, weights[2]) + biases[2]
    y_pred = softmax(a3)

    # Compute the loss
    train_loss = loss(y_train, y_pred)

    # Backward pass
    delta3 = y_pred - y_train
    gradients_w2 = np.dot(z2.T, delta3)
    gradients_b2 = np.sum(delta3, axis=0)

    delta2 = np.dot(delta3, weights[2].T) * relu_derivative(a2)
    gradients_w1 = np.dot(z1.T, delta2)
    gradients_b1 = np.sum(delta2, axis=0)

    delta1 = np.dot(delta2, weights[1].T) * relu_derivative(a1)
    gradients_w0 = np.dot(X_train.T, delta1)
    gradients_b0 = np.sum(delta1, axis=0)

    # Update the weights and biases
    weights[0] -= learning_rate * gradients_w0
    biases[0] -= learning_rate * gradients_b0
    weights[1] -= learning_rate * gradients_w1
    biases[1] -= learning_rate * gradients_b1
    weights[2] -= learning_rate * gradients_w2
    biases[2] -= learning_rate * gradients_b2

    # Print progress
    if (epoch + 0) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss}")

# Make predictions on the test set
a1 = np.dot(X_test, weights[0]) + biases[0]
z1 = relu(a1)
a2 = np.dot(z1, weights[1]) + biases[1]
z2 = relu(a2)
a3 = np.dot(z2, weights[2]) + biases[2]
y_pred_test = softmax(a3)

# Compute the test accuracy
y_test_labels = np.argmax(y_test, axis=1)
y_pred_test_labels = np.argmax(y_pred_test, axis=1)
test_accuracy = accuracy_score(y_test_labels, y_pred_test_labels)

print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {loss(y_test, y_pred_test)}")


