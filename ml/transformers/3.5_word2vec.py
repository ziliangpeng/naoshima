import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import skipgrams
import tensorflow_datasets as tfds

# Load the imdb_reviews dataset
dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data = dataset["train"]

# # Extract the words
# words = []
# for example, label in tfds.as_dataframe(train_data.take(10000), info).iterrows():  # limit to first 10000 reviews for simplicity
#     words.extend(example['text'].split())

# Extract the words
words = []
for text_tensor, _ in train_data.batch(1).take(
    10000
):  # limit to first 10000 reviews for simplicity
    text = text_tensor.numpy()[0].decode("utf-8")  # decode the tensor to get the text
    words.extend(text.split())

# Prepare the data for Word2Vec
vocabulary_size = 10000
window_size = 4
vector_dim = 300
epochs = 5

# Build the dictionary and replace rare words with the UNK token.
counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1
counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
counts = ["UNK"] + [x[0] for x in counts[: vocabulary_size - 1]]
word2id = {word: id for id, word in enumerate(counts)}
id2word = {id: word for id, word in enumerate(counts)}
data = [word2id.get(word, 0) for word in words]

# Generate training data
pairs, labels = skipgrams(
    data, vocabulary_size, window_size=window_size, negative_samples=1.0
)
pair_first_elem = np.array(list(zip(*pairs))[0], dtype="int32")
pair_second_elem = np.array(list(zip(*pairs))[1], dtype="int32")
labels = np.array(labels, dtype="int32")


# Define the model
class Word2Vec(Model):
    def __init__(self, vocabulary_size, vector_dim):
        super(Word2Vec, self).__init__()
        self.embedding = Embedding(
            vocabulary_size, vector_dim, input_length=1, name="embedding"
        )
        self.dot = Dense(1, activation="sigmoid")

    def call(self, pair_first, pair_second):
        first = self.embedding(pair_first)
        second = self.embedding(pair_second)
        dots = self.dot(first * second)
        return dots


# Compile the model
model = Word2Vec(vocabulary_size, vector_dim)
model.compile(loss="binary_crossentropy", optimizer="adam")

# Train the model
model.fit((pair_first_elem, pair_second_elem), labels, epochs=epochs)

# Get the embeddings
embeddings = model.get_layer("embedding").get_weights()[0]

# Validate the result
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

# Print the nearest words
for i in range(valid_size):
    valid_word = id2word[valid_examples[i]]
    nearest = (-similarity[i, :]).numpy().argsort()[1:6]
    log_str = "Nearest to %s:" % valid_word
    for k in range(5):
        close_word = id2word[nearest[k]]
        log_str = "%s %s," % (log_str, close_word)
    print(log_str)
