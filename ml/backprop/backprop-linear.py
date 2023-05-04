"""
Implement back prop for this without using TF.
This is just simple linear regression.

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["accuracy"],
)
"""
# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Define the backpropagation function for Mean Squared Error
def backpropagation_mse(X, y_true, y_pred, a1, z1, a2, z2, weights, biases, learning_rate):
    delta3 = 2 * (y_pred - y_true)
    gradients_w2 = np.dot(z2.T, delta3)
    gradients_b2 = np.sum(delta3, axis=0)

    delta2 = np.dot(delta3, weights[2].T) * relu_derivative(a2)
    gradients_w1 = np.dot(z1.T, delta2)
    gradients_b1 = np.sum(delta2, axis=0)

    delta1 = np.dot(delta2, weights[1].T) * relu_derivative(a1)
    gradients_w0 = np.dot(X.T, delta1)
    gradients_b0 = np.sum(delta1, axis=0)

    # Update the weights and biases
    weights[0] -= learning_rate * gradients_w0
    biases[0] -= learning_rate * gradients_b0
    weights[1] -= learning_rate * gradients_w1
    biases[1] -= learning_rate * gradients_b1
    weights[2] -= learning_rate * gradients_w2
    biases[2] -= learning_rate * gradients_b2

# Gradient descent optimization (backpropagation)
for epoch in range(num_epochs):
    # Forward pass
    y_pred, a1, z1, a2, z2 = forward_pass(X_train, weights, biases)

    # Compute the loss
    train_loss = mse_loss(y_train, y_pred)

    # Backward pass and update weights and biases
    backpropagation_mse(X_train, y_train, y_pred, a1, z1, a2, z2, weights, biases, learning_rate)

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss}")

# Make predictions on the test set
y_pred_test, _, _, _, _ = forward_pass(X_test, weights, biases)

# Compute the test accuracy
y_test_labels = np.argmax(y_test, axis=1)
y_pred_test_labels = np.argmax(y_pred_test, axis=1)
test_accuracy = accuracy_score(y_test_labels, y_pred_test_labels)

print(f"Test accuracy: {test_accuracy}")
