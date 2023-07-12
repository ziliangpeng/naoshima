import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a simple DNN model
model = Sequential(
    [
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(16, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
)

# Train the model using CPU
with tf.device("/GPU:0"):
    history = model.fit(
        x_train, y_train, epochs=5, batch_size=32, validation_split=0.2
    )

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# print weights
for l in model.layers:
    # print("layer:", l)
    if not l.get_weights():
        continue
    weights, bias = l.get_weights()
    print("Learned weight:", weights[0][0])
    print("Learned bias:", bias[0])

# # Plot the training loss and accuracy
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()

# # Visualize example predictions
# num_examples = 10
# predictions = model.predict(x_test[:num_examples])

# plt.figure(figsize=(10, 5))
# for i in range(num_examples):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(x_test[i], cmap='gray')
#     plt.title(f"True: {y_test[i]}\nPred: {np.argmax(predictions[i])}")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()
