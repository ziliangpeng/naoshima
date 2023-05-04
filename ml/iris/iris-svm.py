import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-vs-Rest encoding for multi-class classification
y_train_encoded = np.zeros((len(y_train), 3))
for idx, label in enumerate(y_train):
    y_train_encoded[idx, label] = 1


class SimpleSVM:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        self.weights = np.random.randn(n_features, n_classes)
        self.bias = np.random.randn(1, n_classes)

        for step in range(self.n_iters):
            if step % 50 == 0:
                # Make predictions on the test set
                y_pred = model.predict(X_test)

                # Calculate the accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Accuracy at step {step}: {accuracy:.2f}")

            for idx, x_i in enumerate(X):
                output = np.dot(x_i, self.weights) + self.bias
                y_i = np.argmax(output)
                if y_i != np.argmax(y[idx]):
                    self.weights[:, y_i] -= self.learning_rate * x_i
                    self.weights[:, np.argmax(y[idx])] += self.learning_rate * x_i

    def predict(self, X):
        return np.argmax(np.dot(X, self.weights) + self.bias, axis=1)


# Create the SVM model
model = SimpleSVM(learning_rate=0.01, n_iters=200)

# Train the model
model.fit(X_train, y_train_encoded)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
