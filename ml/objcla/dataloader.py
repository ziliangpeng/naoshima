from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from loguru import logger
from functools import partial


# untested
def load_mnist_from_huggingface():
    tokenizer = AutoTokenizer.from_pretrained("mnist")
    model = AutoModelForSequenceClassification.from_pretrained("mnist")
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

    train_dataset = nlp.tokenizer(["train"], padding=True, truncation=True, return_tensors="pt")
    test_dataset = nlp.tokenizer(["test"], padding=True, truncation=True, return_tensors="pt")

    train_inputs = train_dataset["input_ids"]
    train_labels = train_dataset["attention_mask"]
    test_inputs = test_dataset["input_ids"]
    test_labels = test_dataset["attention_mask"]

    logger.info("Using MNIST dataset from HuggingFace")

    return train_inputs, train_labels, test_inputs, test_labels
    


def _load_keras(loader, name, onehot):
    (x_train, y_train), (x_test, y_test) = loader()
    logger.info(f"Using {name} dataset")

    image_shape = x_train[0].shape
    logger.info(f"The size is {len(x_train)}")
    logger.info(f"The shape is: {image_shape}")

    if onehot:
        y_train = to_categorical(y_train)  # Convert the labels to one-hot encoding
        y_test = to_categorical(y_test)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


load_cifar10 = partial(_load_keras, cifar10.load_data, "cifar10")
load_cifar100 = partial(_load_keras, cifar100.load_data, "cifar100")
load_mnist = partial(_load_keras, mnist.load_data, "mnist")
load_fashion_mnist = partial(_load_keras, fashion_mnist.load_data, "fashion")


if __name__ == "__main__":
    load_mnist(onehot=True)
    load_cifar10(onehot=False)
    X_train, y_train, X_test, y_test = load_mnist(onehot=True)
    logger.info(type(X_train[0])) # type should be numpy.narray
