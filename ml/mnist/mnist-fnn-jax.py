import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

""" Does not work yet. """

# Define the neural network architecture
def init_params(rng):
    w1 = jnp.array(random.normal(rng, (784, 128)))
    b1 = jnp.zeros((128,))
    w2 = jnp.array(random.normal(rng, (128, 64)))
    b2 = jnp.zeros((64,))
    w3 = jnp.array(random.normal(rng, (64, 10)))
    b3 = jnp.zeros((10,))
    return (w1, b1, w2, b2, w3, b3)

def predict(params, x):
    w1, b1, w2, b2, w3, b3 = params
    x = jnp.reshape(x, (x.shape[0], -1))
    hidden1 = jnp.dot(x, w1) + b1
    hidden2 = jnp.dot(jax.nn.relu(hidden1), w2) + b2
    return jnp.dot(jax.nn.relu(hidden2), w3) + b3

# Define the loss function
def loss(params, images, labels):
    preds = predict(params, images)
    return -jnp.mean(jax.nn.log_softmax(preds) * labels)

# Define the accuracy function
def accuracy(params, images, labels):
    preds = predict(params, images)
    return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(labels, axis=1))

# Load the MNIST dataset
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One-hot encode the labels
train_labels = jax.nn.one_hot(train_labels, 10)
test_labels = jax.nn.one_hot(test_labels, 10)

# Initialize the neural network parameters
rng = random.PRNGKey(0)
params = init_params(rng)

# Define the optimizer
@jax.jit
def update(params, x, y, lr):
    grads = jax.grad(loss)(params, x, y)
    return [(param - lr * grad) for param, grad in zip(params, grads)]

# Train the neural network
batch_size = 64
num_epochs = 10
learning_rate = 0.001
num_batches = train_images.shape[0] // batch_size

for epoch in range(num_epochs):
    rng, subrng = random.split(rng)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        x = train_images[start_idx:end_idx]
        y = train_labels[start_idx:end_idx]
        params = update(params, x, y, learning_rate)

    # Evaluate the neural network on the test set
    test_acc = accuracy(params, test_images, test_labels)
    print('Epoch: {} Test accuracy: {:.4f}'.format(epoch, test_acc))
