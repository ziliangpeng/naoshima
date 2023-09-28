from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load dataset
VOCAB_SIZE = 20000
MAX_LENGTH = 250


def load(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    X_train = pad_sequences(
        X_train, maxlen=max_length, padding="post", truncating="post"
    )
    X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")

    return X_train, y_train, X_test, y_test


def info():
    X_train, y_train, X_test, y_test = load()
    print("X_train shape:", X_train.shape)
    print(X_train[0])
    print(y_train[0])


if __name__ == "__main__":
    info()
