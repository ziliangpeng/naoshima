import tensorflow as tf
import numpy as np
import os
import datetime

# Randomized sample data
np.random.seed(42)
X = np.random.rand(1000, 1) * 5
y = 3 * X + 2 + np.random.normal(0, 0.5, size=(1000, 1)) / 10

# Define the neural network model (one-layer DNN)
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["accuracy"],
)
model.summary()

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Create the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, update_freq='epoch')

# Train the model
# Is there a way to visualize how the weights changed?
history = model.fit(X, y, epochs=100, verbose=1, callbacks=[tensorboard_callback])

# To view tensorboard result:
# > tensorboard --logdir logs

# Print the learned weights and bias
weights, bias = model.layers[0].get_weights()
print("Learned weight:", weights[0][0])
print("Learned bias:", bias[0])
