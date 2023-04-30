import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    
    for i in range(num_iterations):
        predictions = X.dot(theta)
        theta -= (1/m) * learning_rate * (X.T.dot(predictions-y))
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history

# Load your data
# X, y = load_your_data()
# Generate random data
num_points = 100
np.random.seed(42)
X = np.random.rand(num_points, 1)
slope = 2
intercept = 3
noise = np.random.randn(num_points, 1) * 0.1
y = slope * X + intercept + noise

# Add the bias term (column of ones) to the X matrix
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize the parameters (theta)
theta = np.zeros(X.shape[1])

# Set the hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

# Print the optimized theta values
print("Optimized theta:", theta)
