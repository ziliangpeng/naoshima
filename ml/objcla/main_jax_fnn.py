import jax
import dataloader
from jax import random
import jax.numpy as jnp
from functools import reduce
import operator
from loguru import logger
import click


def predict1(params, inputs):
    inputs = jnp.reshape(inputs, (inputs.shape[0], -1))

    w1, b1, w2, b2 = params
    num_classes = w2.shape[1]
    logits = jnp.dot(inputs, w1[:, :num_classes]) + b1[:num_classes]
    return logits


def predict2(params, inputs):
    inputs = jnp.reshape(inputs, (inputs.shape[0], -1))

    w1, b1, w2, b2 = params
    hidden1 = jnp.dot(inputs, w1) + b1
    logits = jnp.dot(jax.nn.relu(hidden1), w2) + b2
    return logits


# predict1 is a single layer network, no relu, and thus a linear classifier
# predict2 is a two layer network, with relu, and thus a non-linear classifier
# predict = predict1
predict = predict2


def correct(params, inputs, targets):
    preds = predict(params, inputs)
    return jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(targets, axis=1))


def loss(params, inputs, targets):
    preds = predict(params, inputs)
    l = -jnp.mean(jax.nn.log_softmax(preds) * targets)
    return l


@jax.jit
def update_batch(params, x, y, lr):
    grads = jax.grad(loss)(params, x, y)
    return [(param - lr * grad) for param, grad in zip(params, grads)]


def train_batch(x_train, y_train, x_test, y_test, params, lr):
    batch_size = 128
    num_batches = x_train.shape[0] // batch_size
    epochs = 10
    for epoch in range(epochs):
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            inputs = x_train[start_idx:end_idx]
            labels = y_train[start_idx:end_idx]
            params = update_batch(params, inputs, labels, lr)

        corrects = correct(params, x_train, y_train)
        valid_corrects = correct(params, x_test, y_test)
        logger.info(
            f"Epoch {epoch}, train acc {corrects / x_train.shape[0]:.3f}, valid acc {valid_corrects / x_test.shape[0]:.3f}"
        )


def main():
    # TODO: pass dataset name from cli
    # loader = dataloader.load_cifar10
    loader = dataloader.load_mnist
    x_train, y_train, x_test, y_test = loader(onehot=True)

    num_images = 600  # 6000 * 3
    num_vad = 100
    x_train = x_train[:num_images]
    y_train = y_train[:num_images]
    x_test = x_test[:num_vad]
    y_test = y_test[:num_vad]

    num_classes = y_train.shape[1]

    image_shape = x_train[0].shape
    FC_len = reduce(operator.mul, image_shape)
    logger.info(f"The shape is: {image_shape}, FC len is {FC_len}")

    rng = random.PRNGKey(0)

    def init_params(rng):
        w1 = jnp.array(random.normal(rng, (FC_len, 128)))
        b1 = jnp.zeros((128,))

        w2 = jnp.array(random.normal(rng, (128, num_classes)))
        b2 = jnp.zeros((num_classes,))
        return (w1, b1, w2, b2)

    params = init_params(rng)

    lr = 4e-3
    train_batch(x_train, y_train, x_test, y_test, params, lr)


if __name__ == "__main__":
    main()
