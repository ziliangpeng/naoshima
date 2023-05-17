import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.datasets import imdb

# Load the IMDB dataset
# We'll limit the vocabulary to the top 5000 words, 
# and set the maximum review length to 500 words
vocab_size = 5000
maxlen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size, maxlen=maxlen)

# Load the word index dictionary and create a reverse word index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decoding the reviews back to text
decoded_reviews = lambda x: ' '.join([reverse_word_index.get(i - 3, '?') for i in x])
x_train = [decoded_reviews(x) for x in x_train]
x_test = [decoded_reviews(x) for x in x_test]

# Split the training data to create a validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Use TF-IDF Vectorizer to transform the text data into vectors
vectorizer = TfidfVectorizer(max_features=5000)
x_train = vectorizer.fit_transform(x_train)
x_val = vectorizer.transform(x_val)
x_test = vectorizer.transform(x_test)
print(x_val)

# Train a Logistic Regression model
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

# Evaluate the model
train_preds = model.predict(x_train)
val_preds = model.predict(x_val)
test_preds = model.predict(x_test)

print(f"Train accuracy: {accuracy_score(y_train, train_preds)}")
print(f"Validation accuracy: {accuracy_score(y_val, val_preds)}")
print(f"Test accuracy: {accuracy_score(y_test, test_preds)}")
