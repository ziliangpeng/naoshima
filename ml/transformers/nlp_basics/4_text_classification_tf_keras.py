import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Load the dataset
newsgroups_data = fetch_20newsgroups(subset='all')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups_data.data, newsgroups_data.target, test_size=0.2, random_state=42)
print(X_train[1])
# print(y_train[0])

# Define the parameters for our tokenizer
vocab_size = 10000  # Only consider the top 10k words
max_length = 200  # Only consider the first 200 words of each movie review
trunc_type='post'  # Truncate after the max_length
padding_type='post'  # Pad after the actual text
oov_tok = "<OOV>"  # Out of vocabulary token

# Create the tokenizer and use it to tokenize our texts
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

# Use the tokenizer to convert the sentences into sequences of integers
sequences = tokenizer.texts_to_sequences(X_train)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Do the same for the test data but using the tokenizer fitted on the train data
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(20, activation='softmax')  # 20 for the number of classes in the 20 Newsgroups dataset
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, y_train, epochs=100, validation_data=(test_padded_sequences, y_test))
