import tensorflow as tf
from sklearn.model_selection import train_test_split
import dataloader

X, y = dataloader.generate_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create DNN model for linear regression
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],))])

# Compile the model
model.compile(optimizer="sgd", loss="mean_squared_error")
model.summary()

# Train the model
history = model.fit(
    X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=0
)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")

# Print the learned weights and bias
weights, bias = model.layers[0].get_weights()
print("Learned weight:", weights[0][0])
print("Learned bias:", bias[0])
