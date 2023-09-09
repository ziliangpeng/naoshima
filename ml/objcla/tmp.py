import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.example_libraries import optimizers, stax
from tensorflow.keras.datasets import mnist

""" ConvNet using stax, by GPT"""

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = jnp.array(train_images).reshape(-1, 28, 28, 1).astype("float32") / 255.0
train_labels = jnp.array(train_labels).astype("int32")
test_images = jnp.array(test_images).reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_labels = jnp.array(test_labels).astype("int32")

# Initialize a random key for random number generation
key = random.PRNGKey(0)

# Define the network architecture
init_fun, conv_net = stax.serial(
    stax.Conv(32, (5, 5), (2, 2), padding="SAME"),  # Convolutional layer
    stax.BatchNorm(),  # Batch normalization
    stax.Relu,  # ReLU activation
    stax.Conv(64, (5, 5), (2, 2), padding="SAME"),  # Convolutional layer
    stax.BatchNorm(),  # Batch normalization
    stax.Relu,  # ReLU activation
    stax.Flatten,  # Flatten layer
    stax.Dense(1024),  # Fully connected layer
    stax.Relu,  # ReLU activation
    stax.Dense(10),  # Output layer
)

# Initialize network and optimizer
_, params = init_fun(key, (1, 28, 28, 1))
opt_init, opt_update, get_params = optimizers.adam(1e-4)
state = opt_init(params)

print(params)


# Define the loss function
def loss(params, batch):
    inputs, targets = batch
    preds = conv_net(params, inputs)
    return -jnp.mean(stax.logsoftmax(preds)[jnp.arange(targets.shape[0]), targets])


# Compile the loss function using JIT for better performance
loss = jit(loss)

# Compute the gradient of the loss function
grad_loss = jit(grad(loss))

# Training loop
num_epochs = 10
batch_size = 128

for epoch in range(num_epochs):
    for i in range(0, len(train_images), batch_size):
        key, subkey = random.split(key)
        batch = (train_images[i : i + batch_size], train_labels[i : i + batch_size])
        grads = grad_loss(get_params(state), batch)
        state = opt_update(i, grads, state)

    # Evaluate on test set
    test_loss = loss(get_params(state), (test_images, test_labels))
    print(f"Epoch {epoch+1}, Test Loss: {test_loss}")
