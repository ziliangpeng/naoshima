import time
import jax
import dataloader
from jax import random
import jax.numpy as jnp
from functools import reduce
import operator
from loguru import logger
import click
from jax_pass import init_cnn_params, init_fnn_params, cnn_predict, fnn_predict

"""
TODOs:
- Something is still harcoded, e.g. the image shape, the number of classes, etc. Need to make them parameterizable.
- make it work for other datasets, e.g. cifar10, cifar100, etc.
- run CNN in GPU
"""


predict = None


def correct(params, inputs, targets):
    preds = predict(params, inputs)
    num_correct = jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(targets, axis=1))
    # logger.info(f"num_correct: {num_correct} / {inputs.shape[0]}")
    return num_correct


def loss(params, inputs, targets):
    preds = predict(params, inputs)
    l = -jnp.mean(jax.nn.log_softmax(preds) * targets)
    return l


@jax.jit
def update(params, x, y, lr):
    grads = jax.grad(loss)(params, x, y)
    return [(param - lr * grad) for param, grad in zip(params, grads)]


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


@click.command()
@click.option("--model", default="fnn", help="")
@click.option("--dataset", default="mnist", help="")
def main(model, dataset):
    # TODO: pass dataset name from cli
    # loader = dataloader.load_cifar10
    # loader = dataloader.load_mnist
    loader = getattr(dataloader, f"load_{dataset}")
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

    global predict
    if model == "fnn":
        init_params = init_fnn_params
        predict = fnn_predict
    elif model == "cnn":
        init_params = init_cnn_params
        predict = cnn_predict
    else:
        raise NotImplementedError
    params = init_params(rng)
    logger.info(f"Using {model} model")

    lr = 4e-3
    train(x_train, y_train, x_test, y_test, params, lr)


if __name__ == "__main__":
    main()
