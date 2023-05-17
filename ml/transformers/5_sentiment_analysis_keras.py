import tensorflow as tf
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the IMDB dataset
# We'll limit the vocabulary to the top 5000 words, 
# and set the maximum review length to 500 words
vocab_size = 5000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Make all sequences of the same length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)
