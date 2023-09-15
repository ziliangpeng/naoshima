import time
import jax
import dataloader
from jax import random
import jax.numpy as jnp
from functools import reduce
import operator
from loguru import logger
import click

num_filters = 8


def fast_cnn(params, images, num_filters):
    conv_w, conv_b = params
    conved = jnp.zeros((images.shape[0], 26, 26, num_filters))
    for i in range(26):
        for j in range(26):
            conved = conved.at[:, i, j, :].set(
                jnp.sum(
                    images[:, i : i + 3, j : j + 3, jnp.newaxis] * conv_w[:, :, :],
                    axis=(1, 2),
                )
                + conv_b[:]
            )
    return conved


def predict(params, inputs):
    conv_w, conv_b, w1, b1, w2, b2 = params
    conved = fast_cnn((conv_w, conv_b), inputs, num_filters)

    conved = jnp.reshape(conved, (conved.shape[0], -1))
    conved = jax.nn.relu(conved)

    hidden1 = jnp.dot(conved, w1) + b1
    logits = jnp.dot(jax.nn.relu(hidden1), w2) + b2
    return logits


def correct(params, inputs, targets):
    preds = predict(params, inputs)
    num_correct = jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(targets, axis=1))
    logger.info(f"num_correct: {num_correct} / {inputs.shape[0]}")
    return num_correct


def loss(params, inputs, targets):
    preds = predict(params, inputs)
    l = -jnp.mean(jax.nn.log_softmax(preds) * targets)
    return l


# @jax.jit
def update(params, x, y, lr):
    grads = jax.grad(loss)(params, x, y)
    return [(param - lr * grad) for param, grad in zip(params, grads)]


# @jax.jit
def train(x_train, y_train, x_test, y_test, params, lr):
    batch_size = 128
    num_batches = x_train.shape[0] // batch_size
    epochs = 100
    for epoch in range(epochs):
        start_time = time.time()
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            inputs = x_train[start_idx:end_idx]
            labels = y_train[start_idx:end_idx]
            params = update(params, inputs, labels, lr)

        end_train_time = time.time()
        corrects = correct(params, x_train, y_train)
        valid_corrects = correct(params, x_test, y_test)
        end_time = time.time()
        logger.info(
            f"Epoch {epoch}, train acc {corrects / x_train.shape[0]:.3f}, valid acc {valid_corrects / x_test.shape[0]:.3f}, train time {end_train_time - start_time:.3f}, eval time {end_time - end_train_time:.3f}"
        )


def main():
    # TODO: pass dataset name from cli
    # loader = dataloader.load_cifar10
    loader = dataloader.load_mnist
    x_train, y_train, x_test, y_test = loader(onehot=True)
    # At least it can train and overfit on a small dataset.
    num_images = 600  # 6000 * 3
    num_vad = 100
    x_train = x_train[:num_images]
    y_train = y_train[:num_images]
    x_test = x_test[:num_vad]
    y_test = y_test[:num_vad]
    logger.info(f"training sample {num_images}, validation sample {num_vad}")

    num_classes = y_train.shape[1]

    image_shape = x_train[0].shape
    FC_len = reduce(operator.mul, image_shape)
    logger.info(f"The shape is: {image_shape}, FC len is {FC_len}")

    rng = random.PRNGKey(0)

    def init_params(rng):
        conv_w = jnp.array(random.normal(rng, (3, 3, num_filters)))
        conv_b = jnp.zeros((num_filters,))

        w1 = jnp.array(random.normal(rng, (26 * 26 * num_filters, 128)))
        b1 = jnp.zeros((128,))
        w2 = jnp.array(random.normal(rng, (128, num_classes)))
        b2 = jnp.zeros((num_classes,))
        return (conv_w, conv_b, w1, b1, w2, b2)

    params = init_params(rng)

    lr = 0.01
    train(x_train, y_train, x_test, y_test, params, lr)


if __name__ == "__main__":
    main()
