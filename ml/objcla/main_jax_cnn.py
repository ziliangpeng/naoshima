import time
import jax
import dataloader
from jax import random
import jax.numpy as jnp
from functools import reduce
import operator
from loguru import logger
import click

num_filters = 1


@jax.jit
def predict_batch(params, inputs):
    conv_w, conv_b, w1, b1 = params
    conved = jnp.zeros((inputs.shape[0], 26, 26, num_filters))
    for a in range(inputs.shape[0]):
        for i in range(26):
            for j in range(26):
                for k in range(num_filters):
                    image = inputs[a]
                    # conved = jax.ops.index_update(
                    #     conved, jax.ops.index[i, j], jnp.sum(image[i : i + 3, j : j + 3] * conv_w, axis=(0, 1, 2)) + conv_b
                    # )
                    conved = conved.at[a, i, j, k].set(
                        # jnp.sum(image[i : i + 3, j : j + 3] * conv_w[k], axis=(0, 1, 2)) + conv_b
                        jnp.sum(image[i : i + 3, j : j + 3] * conv_w[:, :, k])
                        + conv_b[k]
                    )
        # logger.info(f'at {a}')

    conved = jnp.reshape(conved, (conved.shape[0], -1))
    logits = jnp.dot(conved, w1) + b1
    # logger.info('done train')
    return logits


# @jax.jit
def correct_batch(params, inputs, targets):
    preds = predict_batch(params, inputs)
    return jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(targets, axis=1))


# @jax.jit
def loss_batch(params, inputs, targets):
    preds = predict_batch(params, inputs)
    l = -jnp.mean(jax.nn.log_softmax(preds) * targets)
    return l


# @jax.jit
def train_batch(x_train, y_train, x_test, y_test, params, lr):
    batch_size = 128
    num_batches = x_train.shape[0] // batch_size
    epochs = 1000
    for epoch in range(epochs):
        start_time = time.time()
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            inputs = x_train[start_idx:end_idx]
            labels = y_train[start_idx:end_idx]
            grads = jax.grad(loss_batch)(params, inputs, labels)
            # logger.info('got grads')
            params = [(param - lr * grad) for param, grad in zip(params, grads)]

        end_train_time = time.time()
        corrects = correct_batch(params, x_train, y_train)
        valid_corrects = correct_batch(params, x_test, y_test)
        end_time = time.time()
        logger.info(
            f"Epoch {epoch}, train acc {corrects / x_train.shape[0]:.3f}, valid acc {valid_corrects / x_test.shape[0]:.3f}, train time {end_train_time - start_time:.3f}, eval time {end_time - end_train_time:.3f}"
        )


def main():
    # TODO: pass dataset name from cli
    # loader = dataloader.load_cifar10
    loader = dataloader.load_mnist
    x_train, y_train, x_test, y_test = loader(onehot=True)
    x_train = x_train[:3]
    y_train = y_train[:3]
    x_test = x_test[:2]
    y_test = y_test[:2]

    num_classes = y_train.shape[1]

    image_shape = x_train[0].shape
    FC_len = reduce(operator.mul, image_shape)
    logger.info(f"The shape is: {image_shape}, FC len is {FC_len}")

    rng = random.PRNGKey(0)

    def init_params(rng):
        conv_w = jnp.array(random.normal(rng, (3, 3, num_filters)))
        conv_b = jnp.zeros((num_filters,))

        w1 = jnp.array(random.normal(rng, (26 * 26 * num_filters, num_classes)))
        b1 = jnp.zeros((num_classes,))
        return (conv_w, conv_b, w1, b1)

    params = init_params(rng)

    lr = 4e-3
    train_batch(x_train, y_train, x_test, y_test, params, lr)


if __name__ == "__main__":
    main()
