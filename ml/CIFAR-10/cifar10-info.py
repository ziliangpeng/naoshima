import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time

# Load the CIFAR-10 dataset
start_time = time.time()
# Load the CIFAR-10 dataset
(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.cifar10.load_data()
end_time = time.time()
loading_time = end_time - start_time
print(f"Time taken to load the CIFAR-10 dataset: {loading_time:.2f} seconds")


# Print the dataset shape
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Define class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Visualize random images from the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    rand_idx = np.random.randint(0, train_images.shape[0])
    plt.imshow(train_images[rand_idx])
    plt.xlabel(class_names[train_labels[rand_idx][0]])
plt.show()

# Calculate class distribution in the dataset
unique, counts = np.unique(train_labels, return_counts=True)
class_distribution = dict(zip(unique, counts))

# Visualize class distribution in a bar plot
plt.figure(figsize=(10, 5))
sns.barplot(
    x=[class_names[i] for i in class_distribution.keys()],
    y=list(class_distribution.values()),
)
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in CIFAR-10 Training Set")
plt.show()

# Calculate the mean and standard deviation of pixel values for each color channel
mean_red = np.mean(train_images[:, :, :, 0])
mean_green = np.mean(train_images[:, :, :, 1])
mean_blue = np.mean(train_images[:, :, :, 2])

std_red = np.std(train_images[:, :, :, 0])
std_green = np.std(train_images[:, :, :, 1])
std_blue = np.std(train_images[:, :, :, 2])

print(f"Mean pixel value (Red channel): {mean_red:.2f}")
print(f"Mean pixel value (Green channel): {mean_green:.2f}")
print(f"Mean pixel value (Blue channel): {mean_blue:.2f}")
print(f"Standard deviation of pixel values (Red channel): {std_red:.2f}")
print(f"Standard deviation of pixel values (Green channel): {std_green:.2f}")
print(f"Standard deviation of pixel values (Blue channel): {std_blue:.2f}")
