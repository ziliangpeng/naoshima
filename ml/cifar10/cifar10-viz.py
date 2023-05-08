import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10 dataset
(X_train, y_train), (_, _) = cifar10.load_data()

# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
)

# Select 10 random images from the dataset
num_images = 10
random_indices = np.random.randint(0, len(X_train), size=num_images)

# Set up the figure for plotting
fig, ax = plt.subplots(num_images, 2, figsize=(10, 30))
fig.tight_layout()

# Plot original and augmented images side-by-side
for i, idx in enumerate(random_indices):
    # Original image
    ax[i, 0].imshow(X_train[idx].astype('uint8'))
    ax[i, 0].set_title('Original Image')
    ax[i, 0].axis('off')

    # Augmented image
    augmented_image = datagen.random_transform(X_train[idx])
    ax[i, 1].imshow(augmented_image.astype('uint8'))
    ax[i, 1].set_title('Augmented Image')
    ax[i, 1].axis('off')

# Display the figure
plt.show()
