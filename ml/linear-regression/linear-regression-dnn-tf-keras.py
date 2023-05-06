import numpy as np
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic regression data
# X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
# Randomized sample data
np.random.seed(42)
X = np.random.rand(1000, 1) * 5
y = 3 * X + 2 + np.random.normal(0, 0.5, size=(1000, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Create DNN model for linear regression
model = tf.keras.Sequential([
    # tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dense(1)

    # try again with simple model
    tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')

# Print the learned weights and bias
weights, bias = model.layers[0].get_weights()
print("Learned weight:", weights[0][0])
print("Learned bias:", bias[0])