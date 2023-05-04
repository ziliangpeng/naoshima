import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    sorted_indices = np.argsort(distances)
    k_nearest_neighbors = y_train[sorted_indices[:k]]
    unique_labels, counts = np.unique(k_nearest_neighbors, return_counts=True)
    return unique_labels[np.argmax(counts)]

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set the number of neighbors (k) to consider
k = 5

# Make predictions on the test set
y_pred = [knn(X_train, y_train, x, k) for x in X_test]

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
