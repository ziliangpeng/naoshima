import dataloader
from tensorflow.keras import layers, models
from loguru import logger
import click


def train_eval(x_train, y_train, x_test, y_test, onehot):
    image_shape = x_train[0].shape
    if onehot:
        num_classes = y_train.shape[1]
    else:
        num_classes = max(y_train) + 1
    logger.info(f"Number of classes: {num_classes}")

    model = models.Sequential(
        [
            layers.Flatten(input_shape=image_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    loss_fn = (
        "categorical_crossentropy" if onehot else "sparse_categorical_crossentropy"
    )
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=4, batch_size=64, validation_split=0.2)


@click.command()
@click.option(
    "--onehot/--no-onehot", default=False, help="Whether to use one-hot encoding or not"
)
def main(onehot):
    # Use the `onehot` variable here to determine whether to use one-hot encoding or not
    print(f"Using one-hot encoding: {onehot}")
    train_eval(*(dataloader.load_mnist(onehot=onehot)), onehot=onehot)
    train_eval(*(dataloader.load_fashion_mnist(onehot=onehot)), onehot=onehot)
    train_eval(*(dataloader.load_cifar10(onehot=onehot)), onehot=onehot)
    train_eval(*(dataloader.load_cifar100(onehot=onehot)), onehot=onehot)


if __name__ == "__main__":
    main()
