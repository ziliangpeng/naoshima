import numpy as np

"""
Bag of Words implemented from scratch.

Note, if implement using scikit-learn:

    from sklearn.feature_extraction.text import CountVectorizer

    corpus = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are great"]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    print('Vocabulary:', vectorizer.get_feature_names_out())
    print('Vectors:', X.toarray())
"""

def get_vocab(corpus):
    vocab = set()
    for text in corpus:
        vocab.update(text.split(' '))
    return sorted(vocab)

def bag_of_words(corpus):
    vocab = get_vocab(corpus)
    vectors = []
    for text in corpus:
        vector = np.zeros(len(vocab), dtype=int)
        for word in text.split(' '):
            index = vocab.index(word)
            vector[index] += 1
        vectors.append(vector)
    return vectors, vocab

corpus = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are great"]
vectors, vocab = bag_of_words(corpus)
print('Vocabulary:', vocab)
# print('Vectors:', vectors)
for v in vectors:
    print(v)
