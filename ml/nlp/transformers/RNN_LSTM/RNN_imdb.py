from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb

# Number of words to consider as features
max_features = 10000

# Cut texts after this number of words 
maxlen = 500

# Load the data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Reverse sequences
X_train = [x[::-1] for x in X_train]
X_test = [x[::-1] for x in X_test]

# Pad sequences
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Define the model
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
