import jax.numpy as jnp
from jax import random, grad, jit, vmap
import dataloader
from dataloader import VOCAB_SIZE, MAX_LENGTH


X_train, y_train, X_test, y_test = dataloader.load()
train_texts = X_train
train_labels = y_train

input_dim = 250
hidden_dim = 64
output_dim = 1

key = random.PRNGKey(0)
params = {
    "w1": random.normal(key, (input_dim, hidden_dim)),
    "b1": jnp.zeros(hidden_dim),
    "w2": random.normal(key, (hidden_dim, output_dim)),
    "b2": jnp.zeros(output_dim),
}


def model(params, x):
    x = jnp.dot(x, params["w1"]) + params["b1"]
    x = jnp.tanh(x)
    x = jnp.dot(x, params["w2"]) + params["b2"]
    return x


def loss(params, x, y):
    preds = model(params, x).squeeze()
    return jnp.mean((preds - y) ** 2)


@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    new_params = {}
    for k in params.keys():
        new_params[k] = params[k] - 0.001 * grads[k]
    return new_params


for epoch in range(420):
    params = update(params, train_texts, train_labels)
    train_loss = loss(params, train_texts, train_labels)
    # print(f"Epoch {epoch+1}, Loss: {train_loss}")

    # Evaluation
    total = 0
    correct = 0
    for data, label in zip(train_texts, train_labels):
        total += 1
        correct += (0 if model(params, data) < 0 else 1) == label
    print(f"Train Accuracy: {correct / total}")
