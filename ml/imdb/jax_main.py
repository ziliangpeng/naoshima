import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import dataloader
from dataloader import VOCAB_SIZE, MAX_LENGTH


X_train, y_train, X_test, y_test = dataloader.load()
train_texts = X_train
train_labels = y_train

input_dim = 250
embedding_dim = 32

key = random.PRNGKey(0)
# simple MLP can achieve good result, and after running long time I can get 80+% accuracy.
# TODO: I need a good optimizer.
params = {
    # Embedding layer
    "w1": random.normal(key, (VOCAB_SIZE, embedding_dim)),
    # MLP
    "w2": random.normal(key, (input_dim * embedding_dim, 32)),
    "b2": jnp.zeros(32),
    "w3": random.normal(key, (32, 1)),
    "b3": jnp.zeros(1),
}


def model(params, x):
    ## Embedding layer.
    # x: (n, length) => (n, length, embedding_dim)
    embedding_matrix = params["w1"]
    x = embedding_matrix[x]
    # print(x.shape)

    ## GlobalAveragePooling1D
    # x = x.mean(axis=1)

    ## Flatten
    x = x.reshape((x.shape[0], -1))
    # print(x.shape)

    ## MLP
    x = jnp.dot(x, params["w2"]) + params["b2"]
    x = jnp.tanh(x)
    x = jnp.dot(x, params["w3"]) + params["b3"]
    x = jax.nn.sigmoid(x)
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


for epoch in range(10000):
    # print(train_texts.shape, train_labels.shape)
    for batch in range(0, len(train_texts), 32):
        params = update(
            params,
            train_texts[batch : batch + 32],
            train_labels[batch : batch + 32],
        )
    # params = update(params, train_texts, train_labels)
    train_loss = loss(params, train_texts, train_labels)
    # print(f"Epoch {epoch+1}, Loss: {train_loss}")

    # Evaluation
    total = 0
    correct = 0
    for data, label in zip(train_texts, train_labels):
        total += 1
        correct += (0 if model(params, data.reshape(1, -1)) < 0.5 else 1) == label
    print(f"Train Accuracy: {correct / total}")
    # print(f"Correct: {correct}, Total: {total}")
