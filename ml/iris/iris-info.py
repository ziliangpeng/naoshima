import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame from the dataset
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# Print basic information about the dataset
print("Feature names:", feature_names)
print("Target names:", target_names)
print("Number of features:", X.shape[1])
print("Number of samples:", X.shape[0])
print("Number of classes:", len(target_names))

# Print descriptive statistics for each feature
print("\nDescriptive statistics for each feature:")
print(df.describe())

# Print the number of samples for each class
print("\nNumber of samples for each class:")
print(df["target"].value_counts())

# Plot histograms for each feature
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(data=df, x=feature_names[0], hue="target", kde=True, ax=axs[0, 0])
sns.histplot(data=df, x=feature_names[1], hue="target", kde=True, ax=axs[0, 1])
sns.histplot(data=df, x=feature_names[2], hue="target", kde=True, ax=axs[1, 0])
sns.histplot(data=df, x=feature_names[3], hue="target", kde=True, ax=axs[1, 1])
plt.tight_layout()
plt.show()

# Plot pairplot for all features
sns.pairplot(df, vars=feature_names, hue="target")
plt.show()
