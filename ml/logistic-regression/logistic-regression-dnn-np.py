import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Preprocess the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the weights and biases
n_features = X_train.shape[1]
w = np.zeros((n_features, 1))
b = 0

# Implement the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Implement the forward propagation algorithm
def forward_propagation(X, w, b):
    z = np.dot(X, w) + b
    a = sigmoid(z)
    return a

# Implement the cost function
def compute_cost(a, y):
    m = y.shape[0]
    cost = (-1/m) * np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    return cost

# Implement the backward propagation algorithm
def backward_propagation(X, a, y):
    m = y.shape[0]
    dz = a - y
    dw = (1/m) * np.dot(X.T, dz)
    db = (1/m) * np.sum(dz)
    return dw, db

# Implement the gradient descent algorithm
def gradient_descent(X_train, y_train, w, b, learning_rate, num_iterations):
    costs = []
    for i in range(num_iterations):
        # Forward propagation
        a = forward_propagation(X_train, w, b)
        
        # Compute cost
        cost = compute_cost(a, y_train)
        costs.append(cost)
        
        # Backward propagation
        dw, db = backward_propagation(X_train, a, y_train)
        
        # Update weights and biases
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Print cost every 100 iterations
        if i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
    
    return w, b, costs

# Train the model
learning_rate = 0.01
num_iterations = 1000
w, b, costs = gradient_descent(X_train, y_train.reshape(-1, 1), w, b, learning_rate, num_iterations)

# Use the trained model to predict the outputs for the test set
y_pred = forward_propagation(X_test, w, b)
y_pred = np.round(y_pred)

# Evaluate the accuracy
accuracy = np.mean(y_pred == y_test.reshape(-1, 1))
print("Accuracy: {}".format(accuracy))
