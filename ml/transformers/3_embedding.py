from gensim.models import Word2Vec

# Define the corpus
corpus = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['the', 'dog', 'sat', 'on', 'the', 'log'],
    ['cats', 'and', 'dogs', 'are', 'great'],
    ['make', 'sure', 'to', 'feed', 'your', 'dog'],
    ['cats', 'are', 'often', 'kept', 'as', 'pets']
]

# Train the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector for a word
vector = model.wv['cat']
print('Vector for "cat":', vector)
