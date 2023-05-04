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
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Initialize Adam parameters
m = 0.0
v = 0.0

# Adam optimization loop
for epoch in range(epochs):
    # Calculate the gradient at the current point
    grad = gradient(x)

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update biased second moment estimate
    v = beta2 * v + (1 - beta2) * (grad**2)

    # Correct bias in first and second moment estimates
    m_hat = m / (1 - beta1 ** (epoch + 1))
    v_hat = v / (1 - beta2 ** (epoch + 1))

    # Update the parameter using the corrected moment estimates
    x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    # Print the progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, x: {x}, Loss: {objective_function(x)}")

# Final result
print(f"Optimized x: {x}, Loss: {objective_function(x)}")
