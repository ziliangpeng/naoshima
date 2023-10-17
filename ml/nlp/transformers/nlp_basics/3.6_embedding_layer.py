import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialize a random embedding matrix
        self.embedding_matrix = np.random.uniform(-1, 1, (vocab_size, embedding_dim))

    def forward(self, input_data):
        # Map each integer in input_data to its embedding vector
        return self.embedding_matrix[input_data]

# Test our embedding layer
vocab_size = 10
embedding_dim = 5
embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)

# Say we have input data [1, 2, 3]
input_data = np.array([1, 2, 3])
print(embedding_layer.forward(input_data))

# NOTE: the embedding matrix is vocab_size x embedding_dim.
# Each vocab has its own array of embedding_dim values.