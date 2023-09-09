import jax
import dataloader
from jax import random
import jax.numpy as jnp
from functools import reduce
import operator
from loguru import logger
import click


def main():
    loader = dataloader.load_mnist
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

    def predict(params, x):
        x = x.ravel()

        w1, b1 = params
        logits = jnp.dot(x, w1) + b1
        return logits

    def correct(params, x, y):
        preds = predict(params, x)
        return jnp.argmax(preds, axis=0) == jnp.argmax(y, axis=0)

    def loss(params, x, y):
        preds = predict(params, x)
        l = -jnp.mean(jax.nn.log_softmax(preds) * y)
        return l

    params = init_params(rng)

    lr = 4e-3
    for epoch in range(10):
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            if i % 10000 == 0:
                logger.info(f"Step {i}, loss {loss(params, x, y)}")
            grads = jax.grad(loss)(params, x, y)
            # for g in grads:
            #     logger.info(str(g))
            params = [(param - lr * grad) for param, grad in zip(params, grads)]

        corrects = 0
        all = 0
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            corrects += correct(params, x, y)
            all += 1
        logger.info(f"Epoch {epoch}, {corrects} correct out of {all}")


if __name__ == "__main__":
    main()
