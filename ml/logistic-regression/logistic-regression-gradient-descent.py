import numpy as np
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

    for _ in range(num_iterations):
        linear_output = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_output)

        dw = np.dot(X.T, (y_pred - y)) / len(y)
        db = np.sum(y_pred - y) / len(y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

data = load_breast_cancer()
X = data.data
y = data.target[:, np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
