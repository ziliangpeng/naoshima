import random
from collections import defaultdict
from nltk.util import ngrams

def train_ngram_model(text, n):
    # Convert text into tokens
    tokens = text.split()
    
    # Create n-grams from tokens
    n_grams = list(ngrams(tokens, n))

    # Create an n-gram language model
    model = defaultdict(lambda: defaultdict(lambda: 0))
    
    for n_gram in n_grams:
        n_gram_till_n_1 = tuple(n_gram[:-1])
        n_gram_n = n_gram[-1]
        model[n_gram_till_n_1][n_gram_n] += 1
    
    # Convert frequency counts to probabilities
    for n_gram_till_n_1 in model:
        total_count = float(sum(model[n_gram_till_n_1].values()))
        for n_gram_n in model[n_gram_till_n_1]:
            model[n_gram_till_n_1][n_gram_n] /= total_count
    
    return model

def generate_sentence(model, start, length):
    current = start
    sentence = current
    
    for i in range(length):
        if current not in model:
            break
        
        possible_words = list(model[current].keys())
        next_word = possible_words[random.randrange(len(possible_words))]
        current = tuple((list(current) + [next_word])[-len(current):])
        sentence += (next_word,)
    
    return sentence

# Your text data here
text = """
This is a simple example of a text. This is another example. And this is yet another example.
"""

# Train the model
n = 3  # you can experiment with different values of n
model = train_ngram_model(text, n)

# Generate a sentence
start = tuple("This is".split())
print(generate_sentence(model, start, 20))
