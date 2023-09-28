import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import dataloader
from dataloader import VOCAB_SIZE, MAX_LENGTH


X_train, y_train, X_test, y_test = dataloader.load()
train_texts = X_train
train_labels = y_train

input_dim = 250
embedding_dim = 128
hidden_dim = 64
output_dim = 1

key = random.PRNGKey(0)
# params = {
#     "w1": random.normal(key, (input_dim, hidden_dim)),
#     "b1": jnp.zeros(hidden_dim),
#     "w2": random.normal(key, (hidden_dim, output_dim)),
#     "b2": jnp.zeros(output_dim),
# }
params = {
    # Embedding layer
    "w1": random.normal(key, (VOCAB_SIZE, embedding_dim)),
    # "b1": jnp.zeros(output_dim),

    "w2": random.normal(key, (MAX_LENGTH * embedding_dim, hidden_dim)),
    "b2": jnp.zeros(hidden_dim),

    "w3": random.normal(key, (hidden_dim, output_dim)),
    "b3": jnp.zeros(output_dim),
}


# def model(params, x):
#     x = jnp.dot(x, params["w1"]) + params["b1"]
#     x = jnp.tanh(x)
#     x = jnp.dot(x, params["w2"]) + params["b2"]
#     return x

def model(params, x):
    # Embedding layer.
    # x: (n, length) => (n, length, embedding_dim)
    embedding_matrix = params["w1"]
    x = embedding_matrix[x]
    # print(x.shape)
    
    x = x.reshape((x.shape[0], -1))
    # print(x.shape)

    # x = jnp.dot(x, params["w1"]) + params["b1"]
    # x = jnp.tanh(x)
    x = jnp.dot(x, params["w2"]) + params["b2"]
    x = jnp.tanh(x)

    x = jnp.dot(x, params["w3"]) + params["b3"]
    return x


def loss(params, x, y):
    preds = model(params, x).squeeze()
    return jnp.mean((preds - y) ** 2)


@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    new_params = {}
    for k in params.keys():
        new_params[k] = params[k] - 0.01 * grads[k]
    return new_params


for epoch in range(42):
    # print(train_texts.shape, train_labels.shape)
    params = update(params, train_texts, train_labels)
    train_loss = loss(params, train_texts, train_labels)
    # print(f"Epoch {epoch+1}, Loss: {train_loss}")

    # Evaluation
    total = 0
    correct = 0
    for data, label in zip(train_texts, train_labels):
        total += 1
        # print(data.shape)
        # print(data.reshape(1, -1).shape)
        correct += (0 if model(params, data.reshape(1, -1)) < 0 else 1) == label
    print(f"Train Accuracy: {correct / total}")
