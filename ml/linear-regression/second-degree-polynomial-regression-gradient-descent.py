import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Gradient of the sigmoid with respect to its input
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Generate some example data
np.random.seed(42)
x = np.linspace(-10, 10, 100)
A = 7
B = 9
C = 10
print(f"y = {A} * x^2 + {B} * x + {C} + noise")
y = A * x**2 + B * x + C  # + np.random.normal(0, 10, 100) / 100

# Transform the input into a 2nd degree polynomial form
X = np.column_stack([x**2, x, np.ones_like(x)])

# Initialize weights
weights = np.random.rand(3)

# Hyperparameters
learning_rate = 0.0001
epochs = 100000

# Gradient Descent
for epoch in range(epochs):
    predictions = X.dot(weights)
    loss = np.mean((predictions - y) ** 2)
    gradients = 2 * X.T.dot(predictions - y) / len(y)
    weights -= learning_rate * gradients
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# # Gradient Descent with Sigmoid Activation
# Sigmoid converges super slow.
# for epoch in range(epochs * 1000):
#     linear_output = X.dot(weights)
#     predictions = sigmoid(linear_output)
#     loss = np.mean((predictions - y)**2)
#     gradients = 2 * X.T.dot((predictions - y) * sigmoid_derivative(linear_output)) / len(y)
#     weights -= learning_rate * gradients
#     if epoch % 1000 == 0:
#         print(f"Epoch {epoch}, Loss: {loss}")

# Final weights represent the coefficients of the polynomial
a, b, c = weights
print(f"Fitted Polynomial: y = {a} * x^2 + {b} * x + {c}")
