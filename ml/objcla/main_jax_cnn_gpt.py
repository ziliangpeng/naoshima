import time
import jax.numpy as jnp
from jax import grad, jit, random, vmap
import tensorflow as tf
from loguru import logger

def conv2d(x, w_filter, b_filter, stride=1):
    n_samples, h_input, w_input, n_channels = x.shape
    f_size, _, input_channels, n_filters = w_filter.shape
    
    h_output = (h_input - f_size) // stride + 1
    w_output = (w_input - f_size) // stride + 1

    x_col = jnp.array([x[:, i:i + f_size, j:j + f_size, :] 
                       for i in range(0, h_input - f_size + 1, stride) 
                       for j in range(0, w_input - f_size + 1, stride)])
    x_col = x_col.reshape(h_output * w_output, x_col.shape[1],  -1)

    w_col = w_filter.reshape(-1, n_filters)
    
    out = x_col.dot(w_col) + b_filter
    out = out.reshape(n_samples, h_output, w_output, n_filters)
    
    return out


# Activation Function (ReLU)
def relu(x):
    return jnp.maximum(0, x)

# Fully Connected Layer
def dense(x, w, b):
    return jnp.dot(x, w) + b

# Forward Pass
def forward(x, params):
    w1, b1, w2, b2, w3, b3 = params
    x = relu(conv2d(x, w1, b1))
    x = x.reshape(x.shape[0], -1)  # Flatten
    x = relu(dense(x, w2, b2))
    x = dense(x, w3, b3)
    return x

# Loss Function (Cross-Entropy Loss)
def loss(params, x, y):
    logits = forward(x, params)
    return -jnp.mean(jnp.sum(logits * y, axis=1) - jnp.logaddexp(0., jnp.sum(jnp.exp(logits), axis=1)))

# Download and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels)

# Initialize random weights
key = random.PRNGKey(0)
w1 = random.normal(key, (5, 5, 1, 8))
b1 = jnp.zeros(8)
w2 = random.normal(key, (8 * 24 * 24, 128))
b2 = jnp.zeros(128)
w3 = random.normal(key, (128, 10))
b3 = jnp.zeros(10)
params = [w1, b1, w2, b2, w3, b3]

# Training Loop
lr = 0.001
for i in range(1000):
    grad_fn = jit(grad(loss))
    gradients = grad_fn(params, jnp.array(train_images[:32]), jnp.array(train_labels[:32]))
    params = [w - lr * dw for w, dw in zip(params, gradients)]
    logger.info(time.time())

# Test the model (use a subset for demonstration)
test_predictions = jit(forward)(jnp.array(test_images[:32]), params)
test_predictions = jnp.argmax(test_predictions, axis=1)
print("Test predictions:", test_predictions)
