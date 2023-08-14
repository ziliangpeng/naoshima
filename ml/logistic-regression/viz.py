import numpy as np
import matplotlib.pyplot as plt


def visualize(X_train, y_train, weights, bias):
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
