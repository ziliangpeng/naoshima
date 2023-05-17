import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Prepare the dataset
data = np.array([i for i in range(1, 101)])  # The numbers from 1 to 100
X = data[:-1].reshape(-1, 1, 1)  # The inputs to the model are the numbers from 1 to 99
y = data[1:]  # The targets are the numbers from 2 to 100

# Define the model
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(1, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, validation_split=0.2)

# Test the model
test_input = np.array([101]).reshape(-1, 1, 1)
test_output = model.predict(test_input)
print(test_output)
