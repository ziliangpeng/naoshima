import tensorflow as tf
import numpy as np

# Randomized sample data
np.random.seed(42)
X = np.random.rand(1000, 1) * 5
y = 3 * X + 2 + np.random.normal(0, 0.5, size=(1000, 1))

# Define the neural network model (one-layer DNN)
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["accuracy"],
)

# Train the model
# Is there a way to visualize how the weights changed?
model.fit(X, y, epochs=100, verbose=1)

# Print the learned weights and bias
weights, bias = model.layers[0].get_weights()
print("Learned weight:", weights[0][0])
print("Learned bias:", bias[0])
print(weights)
print(bias)
