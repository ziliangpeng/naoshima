"""
Implement back prop for this without using TF.
This is just simple linear regression.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],))
])
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Generate synthetic regression data
# X, y = make_regression(n_samples=1000, n_features=1, noise=0.1, random_state=42)
# Randomized sample data
np.random.seed(42)
X = np.random.rand(1000, 1) * 5
y = 3 * X + 2 + np.random.normal(0, 0.5, size=(1000, 1))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # Preprocess data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Initialize weights and bias
np.random.seed(42)
weights = np.random.randn(X_train.shape[1], 1)
bias = np.zeros(1)

# Hyperparameters
learning_rate = 0.01
epochs = 100
batch_size = 32


# Linear regression function
def linear_regression(X, weights, bias):
    return np.dot(X, weights) + bias


# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


# Backpropagation function for Mean Squared Error
def backpropagation_mse(X, y_true, y_pred, weights, bias, learning_rate):
    batch_size = X.shape[0]
    gradients_w = (2 / batch_size) * np.dot(X.T, (y_pred - y_true))
    gradients_b = (2 / batch_size) * np.sum(y_pred - y_true)

    # Update the weights and bias
    weights -= learning_rate * gradients_w
    bias -= learning_rate * gradients_b


# Training loop
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size].reshape(-1, 1)

        y_pred = linear_regression(X_batch, weights, bias)
        loss = mse_loss(y_batch, y_pred)

        backpropagation_mse(X_batch, y_batch, y_pred, weights, bias, learning_rate)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Evaluate the model on the test set
y_test_pred = linear_regression(X_test, weights, bias)
test_loss = mean_squared_error(y_test, y_test_pred)
print(f"Test loss: {test_loss}")
print(weights, bias)
