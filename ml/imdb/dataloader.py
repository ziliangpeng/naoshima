from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


def load(vocab_size, max_length):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    X_train = pad_sequences(
        X_train, maxlen=max_length, padding="post", truncating="post"
    )
    X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")

    return X_train, y_train, X_test, y_test
