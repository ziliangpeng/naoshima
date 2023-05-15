import datetime
import os
import tensorflow as tf

# import numpy as np
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb

from tensorflow.keras.layers import Embedding, Dot, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Define some parameters for your model
VOCAB_SIZE = 10000  # This should be the size of your vocabulary
EMBEDDING_DIM = 300  # This is the dimension of your embedding space
WINDOW_SIZE = 7  # This is the size of the context window


# Create the TensorBoard callback
def make_tb(name):
    prefix = name
    log_dir = os.path.join(
        "logs", prefix + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="batch"
    )


def load_sentences_from_imdb():
    # Load IMDB dataset
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=VOCAB_SIZE
    )
    # print(f"training data size: {len(train_data)}, training label size: {len(train_labels)}")
    # print(f"test data size:     {len(test_data) }, test label size:     {len(test_labels)}")
    train_data = train_data[:1000]

    all_words = set()
    for sentence in train_data:
        for word in sentence:
            all_words.add(word)
    # number of all unique words from all data is 88585
    # number of unique words from 1000 sentences is 11698
    print(f"number of unique words: {len(all_words)}")

    word_index = imdb.get_word_index()
    # We need to add special tokens to the word index
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    # We need to reverse the word index to decode sentences
    reverse_word_index = {v: k for k, v in word_index.items()}
    # Decode the train data
    decoded_sentences = []
    for i in range(len(train_data)):
        decoded_sentences.append(
            " ".join([reverse_word_index.get(i, "?") for i in train_data[i]])
        )
    # for i in range(5):
    #     print(decoded_sentences[i])
    #     print(train_labels[i])
    # TODO: decode test sentences as well
    return decoded_sentences


sentences = load_sentences_from_imdb()

# Tokenize your corpus
# TODO: use tf.keras.layers.TextVectorization instead
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Convert text to sequences of token ids
sequences = tokenizer.texts_to_sequences(sentences)
# print(sentences[0])
# print(sequences[0])
# print(tokenizer.sequences_to_texts([sequences[0]]))


# Flatten the list
sequences = [item for sublist in sequences for item in sublist]
# Generate training data
# This function transforms a sequence of word indexes (list of integers) into tuples of words of the form:
#    - (word, word in the same window), with label 1 (positive samples).
#    - (word, random word from the vocabulary), with label 0 (negative samples).
# Larger window size increase data size.
pairs, labels = skipgrams(
    sequences, vocabulary_size=VOCAB_SIZE, window_size=WINDOW_SIZE
)
print(f"length of pairs: {len(pairs)}")
# for i, p in enumerate(pairs):
#     if p[0] == pairs[0][0]:
#         print(p, labels[i])
print("NOW THE SKIPGRAMS ARE GENERATED")


def build_model(skipgram_pairs):
    # Unpack pairs
    target_words, context_words = zip(*skipgram_pairs)

    # Convert to tensors
    global labels
    target_words = tf.constant(target_words, dtype=tf.int32)
    context_words = tf.constant(context_words, dtype=tf.int32)
    labels = tf.constant(labels, dtype=tf.float32)

    # Define model
    input_target = Input((1,))
    input_context = Input((1,))

    embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=1, name="embedding")

    target = embedding(input_target)
    context = embedding(input_context)

    dot_product = Dot(axes=-1)([target, context])
    dot_product = Flatten()(dot_product)

    word2vec = Model(inputs=[input_target, input_context], outputs=dot_product)

    # Compile model
    word2vec.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )
    return target_words, context_words, word2vec


t_words, c_words, model = build_model(pairs)
model.fit(
    [t_words, c_words],
    labels,
    epochs=10,
    batch_size=512,
    validation_split=0.2,
    callbacks=[make_tb("word2vec")],
)
# test_loss, test_accuracy = model.evaluate([c_words, t_words], labels)
# print(f'Test accuracy: {test_accuracy:.2f}')
