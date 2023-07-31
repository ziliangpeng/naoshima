import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        # Compute the dot product of input and weights, and add bias
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # Apply activation function (sigmoid)
        self.a1 = self.sigmoid(self.z1)
        
        # Compute the dot product of hidden layer and weights, and add bias
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # Apply activation function (linear)
        y_hat = self.z2
        
        return y_hat
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def backward(self, X, y, y_hat, learning_rate):
        # Compute the error between predicted and actual values
        self.error = y_hat - y
        
        # Compute the derivative of the error with respect to the output
        self.delta2 = self.error
        
        # Compute the derivative of the activation function with respect to the input
        self.delta1 = np.dot(self.delta2, self.W2.T) * self.sigmoid_derivative(self.z1)
        
        # Compute the gradients of the weights and biases
        dW2 = np.dot(self.a1.T, self.delta2)
        db2 = np.sum(self.delta2, axis=0, keepdims=True)
        dW1 = np.dot(X.T, self.delta1)
        db1 = np.sum(self.delta1, axis=0)
        
        # Update the weights and biases using the gradients and learning rate
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
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
            print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")
            
    def get_slope(self):
        return self.W2[0][0]
            

def main():
    # Generate some random data
    X = np.random.randn(100, 1)
    y = 3 * X + 4 + np.random.randn(100, 1)

    # Print the input and output data
    print("Input data (X):")
    print(X)
    print("Output data (y):")
    print(y)
    
    # Create a neural network with 1 input, 10 hidden, and 1 output units
    nn = NeuralNetwork(1, 10, 1)
    
    # Train the neural network for 1000 epochs with a learning rate of 0.01
    nn.train(X, y, 100, 0.01)
    
    # Print the final weights and biases
    print("Final weights and biases:")
    print(f"W1: {nn.W1}")
    print(f"b1: {nn.b1}")
    print(f"W2: {nn.W2}")
    print(f"b2: {nn.b2}")
    
    # Get the slope of the line of best fit
    slope = nn.get_slope()
    print(f"Slope: {slope:.4f}")

    
if __name__ == "__main__":
    main()