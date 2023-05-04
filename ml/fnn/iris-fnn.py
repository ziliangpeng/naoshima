import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Preprocess the input features
# Normalize each feature  to the same range.
# Strangely, in practice, not normalizing has better result :/
scaler = StandardScaler()
# comment out this and bunch of things to be simple and comparable to the back prop implementation.
# X = scaler.fit_transform(X)

# Preprocess the output labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the FNN model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(7, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

# Compile the model
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=1, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
