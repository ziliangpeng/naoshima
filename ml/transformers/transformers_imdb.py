from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np

# Number of words to consider as features
max_features = 10000

# Load the data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Convert the integers back to words
word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}
X_train = [" ".join([index_word.get(i - 3, "") for i in x]) for x in X_train]
X_test = [" ".join([index_word.get(i - 3, "") for i in x]) for x in X_test]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Tokenize the inputs
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=500)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=500)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=500)

# Prepare the TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(1000).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(16)

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=3, validation_data=val_dataset)

# Evaluate the model on the test data
model.evaluate(test_dataset)
