import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from collections import Counter

# Load dataset
vocab_size = 10000
max_length = 250

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Get word index and reverse it
word_index = imdb.get_word_index()
print(word_index)
print(len(word_index))
index_word = {i + 3: w for w, i in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"

# Calculate average review length
review_lengths = [len(review) for review in X_train]
avg_review_length = np.mean(review_lengths)
print(f"Average review length: {avg_review_length:.2f}")

# # Plot distribution of review lengths
# plt.hist(review_lengths, bins=50)
# plt.xlabel("Review Length")
# plt.ylabel("Frequency")
# plt.title("Distribution of Review Lengths")
# plt.show()

# Find most frequent words
word_counts = Counter()
for review in X_train:
    for word_index in review:
        word_counts[index_word[word_index]] += 1

most_common_words = word_counts.most_common(20)
print("Most common words:")
for i, (word, count) in enumerate(most_common_words, start=1):
    print(f"{i}. {word}: {count}")
