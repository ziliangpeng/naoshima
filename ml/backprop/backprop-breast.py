"""
Implement back prop for this without using TF

model = tf.keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Preprocess the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a bias term to the input features
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the loss function (binary crossentropy)
def loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialize the weights
weights = np.random.randn(X_train.shape[1])

# Training settings
learning_rate = 0.01
num_epochs = 1000

# Gradient descent optimization (backpropagation)
for epoch in range(num_epochs):
    # Forward pass
    logits = np.dot(X_train, weights)
    y_pred = sigmoid(logits)

    # Compute the gradients
    gradients = np.dot(X_train.T, y_pred - y_train) / X_train.shape[0]

    # Update the weights
    weights -= learning_rate * gradients

    # Compute the loss
    train_loss = loss(y_train, y_pred)

    # Print progress
    if (epoch + 0) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss}")

# Make predictions on the test set
logits = np.dot(X_test, weights)
y_pred_test = sigmoid(logits)
test_loss = loss(y_test, y_pred_test)
y_pred_test = np.round(y_pred_test)

# Compute test accuracy
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")
