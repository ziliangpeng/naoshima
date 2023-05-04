import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_regression(X, y, num_iterations, learning_rate):
    num_features = X.shape[1]
    weights = np.zeros((num_features, 1))
    bias = 0
    # NOTES:
    # weights are a matrix, and bias is a vector. The reason weights are matrix is because there could be multi-output.
    # bias is vector because it represents the relationship between every input and every output.
    # weights is like slope and bias is like inercept.
    # the DNN training process is like human defining the goal to optimize, machine use input to optimize output.

    # When to use Sigmoid and when to use ReLU?
    # Sigmoid:
    #   squashes input into range of [0,1] by a smooth S curve. it is used for binary classification.
    # ReLU:
    #   avoid vanishing gradient problem (make the derivative of very small gradient be 1)

    for _ in range(num_iterations):
        linear_output = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_output)

        dw = np.dot(X.T, (y_pred - y)) / len(y)
        db = np.sum(y_pred - y) / len(y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias


data = load_breast_cancer()
X = data.data[:, :2]  # Select the first two features for visualization
y = data.target[:, np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_iterations = 1000
learning_rate = 0.01

weights, bias = logistic_regression(X_train, y_train, num_iterations, learning_rate)

y_test_pred = sigmoid(np.dot(X_test, weights) + bias)
y_test_pred = (y_test_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")

# Visualization
plt.scatter(
    X_train[y_train.ravel() == 0, 0],
    X_train[y_train.ravel() == 0, 1],
    label="Class 0",
    alpha=0.5,
)
plt.scatter(
    X_train[y_train.ravel() == 1, 0],
    X_train[y_train.ravel() == 1, 1],
    label="Class 1",
    alpha=0.5,
)

x_values = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
y_values = -(bias + weights[0] * x_values) / weights[1]

plt.plot(x_values, y_values, label="Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()