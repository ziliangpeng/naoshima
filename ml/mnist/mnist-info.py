import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = datasets.fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data
y = mnist.target

# Print basic information about the dataset
print("Number of features:", X.shape[1])
print("Number of samples:", X.shape[0])
print("Number of classes:", len(np.unique(y)))

# Create a DataFrame from the dataset
df = pd.DataFrame(X)
df['target'] = y

# Print descriptive statistics for each feature (showing only first 10 features)
print("\nDescriptive statistics for the first 10 features:")
print(df.iloc[:, :10].describe())

# Print the number of samples for each class
print("\nNumber of samples for each class:")
print(df['target'].value_counts())

# Visualize some of the digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    idx = df[df['target'] == str(i)].index[0]
    img = X[idx].reshape(28, 28)
    axes[i].imshow(img, cmap=plt.cm.gray)
    axes[i].set_title(f"Digit: {i}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
