import numpy as np
import dataloader


# This is just a simple Dense layer.
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

    def forward(self, X):
        # Compute the dot product of input and weights, and add bias
        return np.dot(X, self.W1) + self.b1

    def backward(self, X, y, y_hat, learning_rate):
        # Compute the error between predicted and actual values
        self.error = y_hat - y

        # Compute the derivative of the error with respect to the output
        self.delta1 = self.error

        dW1 = np.dot(X.T, self.delta1)
        db1 = np.sum(self.delta1, axis=0)

        # Update the weights and biases using the gradients and learning rate
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            # Forward pass
            y_hat = self.forward(X)

            # Backward pass
            self.backward(X, y, y_hat, learning_rate)

            # Compute and print the loss
            loss = np.mean((y_hat - y) ** 2)
            # print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")

    def get_slope(self):
        return self.W1[0][0]


def main():
    X, y = dataloader.generate_data()

    # Create a neural network with 1 input, 10 hidden, and 1 output units
    nn = NeuralNetwork(1, 1, 1)

    # Train the neural network for 1000 epochs with a learning rate of 0.01
    nn.train(X, y, 10000, 0.0001)

    # Print the final weights and biases
    print("Final weights and biases:")
    print(f"W1: {nn.W1}")
    print(f"b1: {nn.b1}")

    # Get the slope of the line of best fit
    slope = nn.get_slope()
    print(f"Slope: {slope:.4f}")


if __name__ == "__main__":
    main()
