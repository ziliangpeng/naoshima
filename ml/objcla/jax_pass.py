"""
Defines how forward pass is done in JAX.
"""

import jax
import jax.numpy as jnp
from jax import random

"""
CNN trains very slowly and I had to use ~600 training samples.
The performance is thus much worse than FNN.

I tries using only ~600 for FNN, and then CNN is better.
=====
32 filters is giving only ~10+% accuracy.
64 suddenly gives ~70% accuracy at first epoch.
"""
num_filters = 64
num_classes = 10


def init_cnn_params(rng):
    conv_w = jnp.array(random.normal(rng, (3, 3, num_filters)))
    conv_b = jnp.zeros((num_filters,))

    w1 = jnp.array(random.normal(rng, (26 * 26 * num_filters, 128)))
    b1 = jnp.zeros((128,))
    w2 = jnp.array(random.normal(rng, (128, num_classes)))
    b2 = jnp.zeros((num_classes,))
    return (conv_w, conv_b, w1, b1, w2, b2)


def superfast_cnn(params, images, num_filters):
    conv_w, conv_b = params
    conved = jnp.zeros((images.shape[0], 26, 26, num_filters))
    flattened_conv_w = jnp.reshape(conv_w, (9, num_filters))

    tile_Is = jnp.repeat(jnp.arange(3), 3)
    tile_Js = jnp.tile(jnp.arange(3), 3)
    center_Is = jnp.repeat(jnp.arange(26), 26)
    center_Js = jnp.tile(jnp.arange(26), 26)

    Is = jnp.repeat(center_Is, 3 * 3) + jnp.tile(tile_Is, 26 * 26)
    Js = jnp.repeat(center_Js, 3 * 3) + jnp.tile(tile_Js, 26 * 26)
    im2col_tmptmp = images[:, Is, Js]

    im2col = jnp.reshape(im2col_tmptmp, (images.shape[0], 26 * 26, 9))

    conved = im2col @ flattened_conv_w + conv_b
    conved = conved.reshape((images.shape[0], -1, num_filters))
    return conved


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


def cnn_predict(params, inputs):
    conv_w, conv_b, w1, b1, w2, b2 = params
    # conved = fast_cnn((conv_w, conv_b), inputs, num_filters)
    conved = superfast_cnn((conv_w, conv_b), inputs, num_filters)

    conved = jax.nn.relu(conved)
    conved = jnp.reshape(conved, (conved.shape[0], -1))

    hidden1 = jnp.dot(conved, w1) + b1
    logits = jnp.dot(jax.nn.relu(hidden1), w2) + b2
    return logits


def init_fnn_params(rng):
    FC_len = 28 * 28
    w1 = jnp.array(random.normal(rng, (FC_len, 128)))
    b1 = jnp.zeros((128,))

    w2 = jnp.array(random.normal(rng, (128, num_classes)))
    b2 = jnp.zeros((num_classes,))
    return (w1, b1, w2, b2)


def fnn_predict(params, inputs):
    inputs = jnp.reshape(inputs, (inputs.shape[0], -1))

    w1, b1, w2, b2 = params
    hidden1 = jnp.dot(inputs, w1) + b1
    logits = jnp.dot(jax.nn.relu(hidden1), w2) + b2
    return logits
