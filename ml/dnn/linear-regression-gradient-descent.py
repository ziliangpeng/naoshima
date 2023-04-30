import numpy as np
import matplotlib.pyplot as plt
import imageio

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    theta_history = []
    
    for i in range(num_iterations):
        predictions = X.dot(theta)
        # So here's how I understand it:
        # X is a Nx2 matrix, first colum all 1, second column all x.
        # theta is 2x1 (intercept, slope)
        #   theta is a matrix that represents 2 parameters.
        # y is Nx1 matrix.
        # X.dot(theta) gets a 2x1 predictions, basically (1 * theta[0], x * theta[1])
        #
        # so this is optimizing 2 parameters at the same time.
        # - sum of y diff minimize - find intercept
        # - weighted y diff minimize - find slope
        #
        # in DNN world, the `theta -=` part is the back propagation, basically updating weights.
        # there is no activation function, since there's no need for non-linearity, or an identity function
        # we can think of the activation function as 
        # ```
        #     def identity(x):
        #         return x
        #     predictions = identity(X.dot(theta))
        # ```
        # in this case, activtion function is used for forward calculation, not backward calculation?
        # so... if we use ReLU as activation function, the effect is that `y = ReLU(Wx + b)`
        #
        # GPT-4 says activation function is used in both forward and back prop, 
        # in forward pass, act func is used normally.
        # in the back prop, a derivative(导数) of act func (e.g. dReLU) is used.
        # ReLU(x) = max(0, x)
        # dReLU(x)/dx = { 1, if x > 0; 0, if x < 0 }
        # so every act func needs to be derivable/differentiable (可导)
        # Oh.. so derivative means how much y changes over x. so it makes sense to use d for backward calculation.

        theta -= (1/m) * learning_rate * (X.T.dot(predictions-y))
        theta_history.append(theta.copy())
        # if i == 1 or i == num_iterations - 1:
            # print("inspecting theta")
            # print(X)
            # print(y)
            # print(X.T.dot(predictions-y))
            # print((1/m) * learning_rate * (X.T.dot(predictions-y)))
            # print(theta)
            # print("done inspecting theta")
        if i % 1000 == 0:
            print(theta)
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history, theta_history

# Generate random data
num_points = 1000
np.random.seed(42)
x = np.random.rand(num_points, 1)
slope = 2
intercept = 3
noise = np.random.randn(num_points, 1) * 0.1
y = slope * x + intercept + noise

# Add the bias term (column of ones) to the X matrix
X = np.hstack((np.ones((x.shape[0], 1)), x))

# Initialize the parameters (theta)
theta = np.zeros((X.shape[1], 1))

# Set the hyperparameters
# NOTE: yeah, so a learning of 2 messes up the learning lol
learning_rate = 0.1
num_iterations = 200

# Perform gradient descent
theta, cost_history, theta_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

# Print the optimized theta values
print("slope: %d, intercept %d" % (slope, intercept))
print("Optimized theta:", theta)


# print(theta_history)
for i, t in enumerate(theta_history):
    plt.scatter(x, y)
    plt.plot(x, X.dot(t), color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Iteration {i+1}")
    plt.savefig(f"lrgs/iteration_{i+1}.png")
    plt.clf()
with imageio.get_writer('lrgs/gradient_descent_optimization.mp4', mode='I', fps=32) as writer:
    for i in range(num_iterations):
        image = imageio.imread(f"lrgs/iteration_{i+1}.png")
        writer.append_data(image)
