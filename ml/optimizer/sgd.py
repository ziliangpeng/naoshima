import numpy as np


# Objective function: f(x) = x^2
def objective_function(x):
    return x**2


# Gradient of the objective function: f'(x) = 2x
def gradient(x):
    return 2 * x


# Initialize parameters
x = np.random.randn()
learning_rate = 0.01
epochs = 100

# SGD optimization loop
for epoch in range(epochs):
    # Calculate the gradient at the current point
    grad = gradient(x)

    # Update the parameter using the gradient
    x = x - learning_rate * grad

    # Print the progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, x: {x}, Loss: {objective_function(x)}")

# Final result
print(f"Optimized x: {x}, Loss: {objective_function(x)}")
