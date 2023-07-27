import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

"""
Assignment:

Implement a sentiment analysis model using a Recurrent Neural Network in Python and TensorFlow.
Your model should take as input a sequence of words (a sentence) and output whether the sentiment
of the sentence is positive or negative. You can use the IMDb movie reviews dataset which is a
binary sentiment analysis dataset available in TensorFlow datasets.
"""

# Load the IMDB dataset
(train_dataset, test_dataset), dataset_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)

# Print the dataset info
print(dataset_info)

# Prepare the data for training
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = (
    train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build a subwords tokenizer from the training dataset
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (review.numpy() for review, _ in train_dataset), target_vocab_size=2**13
)


# Define a function to encode the reviews
def encode(review, label):
    encoded_review = tokenizer.encode(review.numpy())
    return encoded_review, label


# Apply the encoding function to the datasets
train_dataset = train_dataset.map(
    lambda review, label: tf.py_function(encode, [review, label], [tf.int64, tf.int64])
)
test_dataset = test_dataset.map(
    lambda review, label: tf.py_function(encode, [review, label], [tf.int64, tf.int64])
)

# Pad the sequences to the same length
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_dataset.padded_batch(BATCH_SIZE)

# Define the RNN model
model = keras.Sequential(
    [
        layers.Embedding(tokenizer.vocab_size, 64),
        layers.Bidirectional(layers.SimpleRNN(64, return_sequences=False)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ]
)

# Compile the model
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
