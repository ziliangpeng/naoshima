import datetime
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb

from tensorflow.keras.layers import Embedding, Dot, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


USE_GENERATOR = False

# Create the TensorBoard callback
def make_tb(name):
    prefix = name
    log_dir = os.path.join(
        "logs", prefix + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq='batch'
    )

# Define some parameters for your model
vocab_size = 15000  # This should be the size of your vocabulary
embedding_dim = 400  # This is the dimension of your embedding space
window_size = 2  # This is the size of the context window

# Load IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

train_data = train_data[:10000]
train_labels = train_labels[:10000]
test_data = test_data[:10000]
test_labels = test_labels[:10000]

# print(len(train_data), len(train_labels))
# print(len(test_data), len(test_labels))

# Get word index from IMDB dataset
word_index = imdb.get_word_index()

# We need to add special tokens to the word index
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# We need to reverse the word index to decode sentences
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decode the train data
decoded_sentences = []
for i in range(len(train_data)):
    decoded_sentences.append(' '.join([reverse_word_index.get(i, '?') for i in train_data[i]]))

# Tokenize your corpus
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(decoded_sentences)

# Convert text to sequences of token ids
sequences = tokenizer.texts_to_sequences(decoded_sentences)

print("now the data is tokenized")

def generate_skipgrams(sequences, vocabulary_size, window_size):
    for sequence in sequences:
        print(sequence)
        pairs, labels = skipgrams(sequence, vocabulary_size, window_size)
        if pairs:
            pairs = np.array(pairs, dtype='int32')
            labels = np.array(labels, dtype='int32')
            yield ([pairs[:, 0], pairs[:, 1]], labels)


if USE_GENERATOR:
    # Create generator
    generator = generate_skipgrams(sequences, vocab_size, window_size)
else:
    # Flatten the list
    sequences = [item for sublist in sequences for item in sublist]

    # Generate training data
    pairs, labels = skipgrams(sequences, vocabulary_size=vocab_size, window_size=window_size)

print("now the data is generated")
# print(type(labels))
# print(type(pairs))
# print(labels[0])
# print(pairs[0])


# # Define your model
# class Word2Vec(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_dim):
#         super(Word2Vec, self).__init__()
#         self.target_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=1)
#         self.context_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=1)
#         self.dots = tf.keras.layers.Dot(axes=(3, 2))
#         self.flatten = tf.keras.layers.Flatten()

#     def call(self, pair_batch):
#         # target, context = pair
#         # we = self.target_embedding(target)
#         # ce = self.context_embedding(context)
#         # dots = self.dots([we, ce])
#         # return self.flatten(dots)

#         # GPT says this tries not to iterate over a tensor

#         for pair in pair_batch:

#             print(pair.shape)
#             print(type(pair))
#             print(pair)
#             print("length:", len(pair))
#             print(dir(pair))
#             target, context = pair[0], pair[1]
#             we = self.target_embedding(target)
#             ce = self.context_embedding(context)
#             dots = self.dots([we, ce])
#             return self.flatten(dots)

# # Instantiate your model
# model = Word2Vec(vocab_size, embedding_dim)

# Rewrite model using Keras
# Unpack pairs
target_words, context_words = zip(*pairs)

# Convert to tensors
target_words = tf.constant(target_words, dtype=tf.int32)
context_words = tf.constant(context_words, dtype=tf.int32)
labels = tf.constant(labels, dtype=tf.float32)

# Define model
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, embedding_dim, input_length=1, name='embedding')

target = embedding(input_target)
context = embedding(input_context)

dot_product = Dot(axes=-1)([target, context])
dot_product = Flatten()(dot_product)

word2vec = Model(inputs=[input_target, input_context], outputs=dot_product)

# Compile model
word2vec.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

# Train model
word2vec.fit([target_words, context_words], labels, epochs=10, batch_size=512, validation_split=0.2, callbacks=[make_tb("word2vec")])


# # Compile your model
# model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
# print("now the model is defined")

# print("now the training starts")
# if USE_GENERATOR:
#     # # Train model using generator
#     print("using generator")
#     model.fit_generator(generator, steps_per_epoch=500, epochs=10)
# else:
#     # Train your model
#     print("not using generator")
#     model.fit(pairs, labels, epochs=10, batch_size=1, verbose=1, callbacks=[make_tb("word2vec")])
#     # model.fit(pairs, labels, epochs=10, verbose=1, callbacks=[make_tb("word2vec")])
#     # Train your model
#     # model.fit([target_words, context_words], labels, epochs=10, batch_size=512)