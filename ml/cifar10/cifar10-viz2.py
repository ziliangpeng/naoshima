import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Select a specific class to visualize (0-9)
selected_class = 8
#
# 0 airplane
# 1 automobile
# 2 bird
# 3 cat
# deer
# 5 dog
# frog
# 7 horse
# ship
# 9 truck

# Get indices of the selected class images in the training and validation sets
train_indices = np.where(y_train == selected_class)[0]
test_indices = np.where(y_test == selected_class)[0]

# Create a function to display images in a grid
def display_images(images, title):
    images = images[:300]
    all_images = images
    for s in range(len(images)//100):
        images = all_images[s*100:s*100+100]
        plt.figure(figsize=(10, 10))
        columns = 10
        rows = int(np.ceil(len(images) / columns))
        for i, image in enumerate(images):
            plt.subplot(rows, columns, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image)
        plt.suptitle(title, fontsize=16)
        plt.show()

# Display the selected class images from the training set
display_images(x_train[train_indices], f"Class {selected_class} (Training Set)")

# Display the selected class images from the validation set
display_images(x_test[test_indices], f"Class {selected_class} (Validation Set)")
