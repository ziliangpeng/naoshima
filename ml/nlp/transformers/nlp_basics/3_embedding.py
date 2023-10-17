import os
import time
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Define the corpus
corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["cats", "and", "dogs", "are", "great"],
    ["make", "sure", "to", "feed", "your", "dog"],
    ["cats", "are", "often", "kept", "as", "pets"],
]

"""
Note: To run this code, you'll need a corpus to train the Word2Vec embeddings on. For simplicity, I'll use a list of sentences (each sentence is a list of words), but in a real-world situation, you might use a larger corpus, like a collection of documents or an entire dataset.

Note: embedding is to represent word as values. there are many apporaches to do this. Word2Vec and GloVe are two popular approaches.

Word2Vec is to train a neural network to predict the center word given the context of the word, using sliding window and predict the center word. GloVe is to train a neural network to predict the probability that two words appear together.

The output layer of the neural network is the word embedding.
"""


def train_word2vec():
    """Training Word2Vec"""

    # Train the Word2Vec model
    model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

    # Get the vector for a word
    vector = model.wv["cat"]
    print('Vector for "cat":', vector)


WORD2VEC_OUTPUT_FILE = "glove.6B.100d.txt.word2vec"  # output file in Word2Vec format


def load_pretrained_glove():
    """
    To load GloVe embeddings, you first need to convert them to Word2Vec format. You can do this using the glove2word2vec script provided by gensim. Here's how to do it:
    """
    # File paths
    # Note: need to download GloVe embeddings from https://nlp.stanford.edu/projects/glove/
    # https://nlp.stanford.edu/data/glove.6B.zip
    glove_file = "glove.6B.100d.txt"  # path to the GloVe file

    # Convert the GloVe file to Word2Vec format
    glove2word2vec(glove_file, WORD2VEC_OUTPUT_FILE)


def load_word2vec_formatted_glove():
    """Now you can load the Word2Vec-formatted GloVe embeddings like this"""

    # Load the Word2Vec-formatted GloVe embeddings
    glove_model = KeyedVectors.load_word2vec_format(WORD2VEC_OUTPUT_FILE, binary=False)

    # Get the vector for a word
    w = "cat"
    vector = glove_model[w]
    print(f'Vector for "{w}": {vector}')

    w = "dog"
    vector = glove_model[w]
    print(f'Vector for "{w}": {vector}')


def main():
    # Try training from scratch.
    train_word2vec()

    # Try loading pretrained GloVe embeddings.
    if not os.path.exists(WORD2VEC_OUTPUT_FILE):
        start_time = time.time()
        load_pretrained_glove()
        print("Time taken:", time.time() - start_time)

    # Try loading Word2Vec-formatted GloVe embeddings.
    load_word2vec_formatted_glove()


if __name__ == "__main__":
    main()
