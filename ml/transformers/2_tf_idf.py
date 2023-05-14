import numpy as np
import math

"""
TF-IDF implemented from scratch.

Note: if implements using scikit-learn:

    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are great"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print('Vocabulary:', vectorizer.get_feature_names_out())
    print('Vectors:', X.toarray())
"""

def get_vocab(corpus):
    vocab = set()
    for text in corpus:
        vocab.update(text.split(' '))
    return sorted(vocab)

def tf_idf(corpus):
    vocab = get_vocab(corpus)
    vectors = []
    N = len(corpus)

    # calculate IDF
    idf = np.zeros(len(vocab), dtype=float)
    for text in corpus:
        for word in set(text.split(' ')):
            index = vocab.index(word)
            idf[index] += 1
    idf = np.log(N / idf)

    # calculate TF and TF-IDF
    for text in corpus:
        vector = np.zeros(len(vocab), dtype=float)
        words = text.split(' ')
        for word in words:
            index = vocab.index(word)
            vector[index] += 1
        vector /= len(words)
        vector *= idf
        vectors.append(vector)

    return vectors, vocab

corpus = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are great"]
vectors, vocab = tf_idf(corpus)
print('Vocabulary:', vocab)
# print('Vectors:', vectors)
for v in vectors:
    print(v)
    print()
