import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for _ in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, h - y)
        theta -= alpha * gradient
    return theta


def one_vs_rest(X, y, num_classes, alpha, num_iters):
    thetas = []
    for c in range(num_classes):
        y_c = np.where(y == c, 1, 0)
        theta_c = np.zeros((X.shape[1], 1))
        theta_c = gradient_descent(X, y_c, theta_c, alpha, num_iters)
        thetas.append(theta_c)
    return np.hstack(thetas)


# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add the bias term (column of ones) to the input features
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Initialize the learning rate and number of iterations
alpha = 0.1
num_iters = 1000

# Train the logistic regression model using one-vs-rest approach
num_classes = len(np.unique(y))
thetas = one_vs_rest(X_train, y_train, num_classes, alpha, num_iters)

# Make predictions on the test set
y_pred_probs = sigmoid(np.dot(X_test, thetas))
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
