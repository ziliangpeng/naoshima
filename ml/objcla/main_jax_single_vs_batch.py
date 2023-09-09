import jax
import dataloader
from jax import random
import jax.numpy as jnp
from functools import reduce
import operator
from loguru import logger
import click


def predict(params, x):
    # Originally we can use x.ravel() to flatten x.
    # here, we can use jnp.reshape to flatten x while preserving the number of examples if shape[0] is example dimension.
    x = jnp.reshape(x, (-1))

    w1, b1 = params
    logits = jnp.dot(x, w1) + b1
    return logits


def predict_batch(params, inputs):
    inputs = jnp.reshape(inputs, (inputs.shape[0], -1))

    w1, b1 = params
    logits = jnp.dot(inputs, w1) + b1
    return logits


def correct(params, x, y):
    preds = predict(params, x)
    return jnp.argmax(preds, axis=0) == jnp.argmax(y, axis=0)


def correct_batch(params, inputs, targets):
    preds = predict_batch(params, inputs)
    return jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(targets, axis=1))


def loss(params, x, y):
    preds = predict(params, x)
    l = -jnp.mean(jax.nn.log_softmax(preds) * y)
    return l


def loss_batch(params, inputs, targets):
    preds = predict_batch(params, inputs)
    l = -jnp.mean(jax.nn.log_softmax(preds) * targets)
    return l


def train_single(x_train, y_train, x_test, y_test, params, lr):
    for epoch in range(10):
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            grads = jax.grad(loss)(params, x, y)
            params = [(param - lr * grad) for param, grad in zip(params, grads)]

        corrects = 0
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            corrects += correct(params, x, y)
        valid_corrects = 0
        for i, (x, y) in enumerate(zip(x_test, y_test)):
            valid_corrects += correct(params, x, y)
        logger.info(
            f"Epoch {epoch}, train acc {corrects / x_train.shape[0]:.3f}, valid acc {valid_corrects / x_test.shape[0]:.3f}"
        )


def train_batch(x_train, y_train, x_test, y_test, params, lr):
    batch_size = 128
    num_batches = x_train.shape[0] // batch_size
    epochs = 1000
    for epoch in range(epochs):
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            inputs = x_train[start_idx:end_idx]
            labels = y_train[start_idx:end_idx]
            grads = jax.grad(loss_batch)(params, inputs, labels)
            params = [(param - lr * grad) for param, grad in zip(params, grads)]

        corrects = correct_batch(params, x_train, y_train)
        valid_corrects = correct_batch(params, x_test, y_test)
        logger.info(
            f"Epoch {epoch}, train acc {corrects / x_train.shape[0]:.3f}, valid acc {valid_corrects / x_test.shape[0]:.3f}"
        )


def main():
    # TODO: pass dataset name from cli
    loader = dataloader.load_cifar10
    # loader = dataloader.load_mnist
    x_train, y_train, x_test, y_test = loader(onehot=True)

    num_classes = y_train.shape[1]

    image_shape = x_train[0].shape
    FC_len = reduce(operator.mul, image_shape)
    logger.info(f"The shape is: {image_shape}, FC len is {FC_len}")

    rng = random.PRNGKey(0)

    def init_params(rng):
        w1 = jnp.array(random.normal(rng, (FC_len, num_classes)))
        b1 = jnp.zeros((num_classes,))
        return (w1, b1)

    params = init_params(rng)

    lr = 4e-3
    train_single(x_train, y_train, x_test, y_test, params, lr)
    # train_batch(x_train, y_train, x_test, y_test, params, lr)


if __name__ == "__main__":
    main()
