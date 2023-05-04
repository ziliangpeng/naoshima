import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import datetime

# Load a dataset suitable for binary classification
# In this example, we'll use the breast cancer dataset from scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the logistic regression model
model = tf.keras.Sequential([
    # Original NN:
    layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))

    # Experiment NN:
    # tf.keras.layers.Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
    # layers.Dense(1, activation='sigmoid', input_shape=(10,))
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Create the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, update_freq='epoch')

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[tensorboard_callback])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# To view tensorboard result:
# > tensorboard --logdir logs
