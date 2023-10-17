import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

"""
Tries to predict sotck price, which is a time series data, using RNN.

It is one way to generating sequence.

However, predicting stock price is very difficult, because it is very noisy."""

# Download historical stock price data
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')['Close'].values

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Prepare the dataset
lookback = 5  # The number of previous days to use to predict the next day
X = np.array([data[i:i+lookback].flatten() for i in range(len(data)-lookback)])
y = data[lookback:]

# Split the data into training and validation sets
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Reshape the inputs to be suitable for the SimpleRNN layer
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# Define the model
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(lookback, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
