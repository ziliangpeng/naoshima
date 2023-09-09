import tensorflow as tf
import numpy as np


""" Does not work"""

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert the labels to one-hot encoding
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# Define hyperparameters
learning_rate = 0.1
batch_size = 128
num_steps = len(x_train) // batch_size
n_hidden_1 = 256

# Define the model
class SimpleFNN(tf.Module):
    def __init__(self):
        # self.w1 = tf.Variable(tf.random.normal([784, n_hidden_1]), trainable=True)
        # self.b1 = tf.Variable(tf.random.normal([n_hidden_1]), trainable=True)
        # self.w2 = tf.Variable(tf.random.normal([n_hidden_1, 10]), trainable=True)
        # self.b2 = tf.Variable(tf.random.normal([10]), trainable=True)

        self.w1 = tf.Variable(tf.random.normal([784, n_hidden_1], dtype=tf.float64), trainable=True)
        self.b1 = tf.Variable(tf.random.normal([n_hidden_1], dtype=tf.float64), trainable=True)
        self.w2 = tf.Variable(tf.random.normal([n_hidden_1, 10], dtype=tf.float64), trainable=True)
        self.b2 = tf.Variable(tf.random.normal([10], dtype=tf.float64), trainable=True)


    def __call__(self, x):
        print(x[0][0].dtype)
        print(self.w1[0][0].dtype)
        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.w2) + self.b2
        return x

# Loss function
def loss_fn(model, x, y):
    logits = model(x)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# Optimization step
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x, y)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Define optimizer
optimizer = tf.optimizers.Adam(learning_rate)

# Initialize the model
model = SimpleFNN()

# Training loop
for step in range(num_steps):
    batch_x = x_train[step * batch_size: (step + 1) * batch_size]
    batch_y = y_train[step * batch_size: (step + 1) * batch_size]
    loss = train_step(model, batch_x, batch_y)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss}")

# Evaluate the model
logits = model(x_test)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print(f"Testing Accuracy: {accuracy}")
