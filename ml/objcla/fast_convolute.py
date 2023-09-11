import sys
import jax.numpy as jnp
from jax import random

import dataloader
import time

from loguru import logger


def timeit(func, verbose=False):
    def wrapper(*args):
        start = time.time()
        ret = func(*args)
        end = time.time()
        if verbose:
            logger.info(f"Time taken for {func.__name__}: {end - start}")
        return end - start, ret

    return wrapper


@timeit
def for4(params, inputs, num_filters):
    conv_w, conv_b = params
    conved = jnp.zeros((inputs.shape[0], 26, 26, num_filters))
    for a in range(inputs.shape[0]):
        for i in range(26):
            for j in range(26):
                for k in range(num_filters):
                    image = inputs[a]
                    conved = conved.at[a, i, j, k].set(
                        jnp.sum(image[i : i + 3, j : j + 3] * conv_w[:, :, k])
                        + conv_b[k]
                    )
    return conved


@timeit
def for3(params, inputs, num_filters):
    conv_w, conv_b = params
    conved = jnp.zeros((inputs.shape[0], 26, 26, num_filters))
    for a in range(inputs.shape[0]):
        for i in range(26):
            for j in range(26):
                image = inputs[a]
                conved = conved.at[a, i, j, :].set(
                    jnp.sum(
                        image[i : i + 3, j : j + 3, jnp.newaxis] * conv_w[:, :, :],
                        axis=(0, 1),
                    )
                    + conv_b[:]
                )
    return conved


@timeit
def for2(params, images, num_filters):
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


def test_scalability(full_images):
    rng = random.PRNGKey(0)
    for num_images in [1, 2, 4, 8, 16, 32]:
        for num_filters in [1, 2, 4, 8, 16, 32]:
            logger.info(f"num_images: {num_images}, num_filters: {num_filters}")

            images = full_images[:num_images]

            conv_w = jnp.array(random.normal(rng, (3, 3, num_filters)))
            conv_b = jnp.zeros((num_filters,))
            params = (conv_w, conv_b)
            time4, ret4 = for4(params, images, num_filters)
            # time3, ret3 = for3(params, images, num_filters)
            time2, ret2 = for2(params, images, num_filters)
            logger.info(f"Time ratio: {time4 / time2}")
            logger.info(f"per kernel ratio: {time4 / (time2) / num_filters / 2}")
            logger.info(
                f"per kernel image ratio: {time4 / (time2) / num_filters / num_images / 2}"
            )


def test_parallelism(full_images):
    rng = random.PRNGKey(0)

    num_filters = 1
    num_images = 1
    conv_w = jnp.array(random.normal(rng, (3, 3, num_filters)))
    conv_b = jnp.zeros((num_filters,))
    params = (conv_w, conv_b)
    images = full_images[:num_images]
    unit_time, _ = for2(params, images, 1)
    logger.info(f"unit_time: {unit_time}")

    for num_images in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for num_filters in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            conv_w = jnp.array(random.normal(rng, (3, 3, num_filters)))
            conv_b = jnp.zeros((num_filters,))
            params = (conv_w, conv_b)

            images = full_images[:num_images]
            time2, _ = for2(params, images, num_filters)
            logger.info(
                f"num_images: {num_images}, num_filters: {num_filters}, time ratio: {time2 / unit_time}"
            )


def test_accuracy(full_images):
    rng = random.PRNGKey(0)
    num_filters = 8
    num_images = 9
    logger.info(
        f"Testing accuracy with num_images: {num_images}, num_filters: {num_filters}"
    )

    conv_w = jnp.array(random.normal(rng, (3, 3, num_filters)))
    conv_b = jnp.zeros((num_filters,))
    params = (conv_w, conv_b)
    images = full_images[:num_images]

    time4, ret4 = for4(params, images, num_filters)
    time3, ret3 = for3(params, images, num_filters)
    time2, ret2 = for2(params, images, num_filters)

    assert jnp.allclose(ret2, ret3)
    assert jnp.allclose(ret2, ret4)
    assert jnp.allclose(ret3, ret4)
    logger.info("All close!")


if __name__ == "__main__":
    NUM_IMAGES = 3
    NUM_FILTERS = 3

    loader = dataloader.load_mnist
    images, _, _, _ = loader(onehot=True)

    # test_scalability(images)
    # test_accuracy(images)
    test_parallelism(images)
