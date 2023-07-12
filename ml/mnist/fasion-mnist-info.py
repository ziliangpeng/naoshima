# import necessary packages
import tensorflow as tf
import matplotlib.pyplot as plt

# load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# class names (as they are not included with the dataset)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print basic info
print("Train images shape:", train_images.shape)
print("Number of train labels:", len(train_labels))
print("Test images shape:", test_images.shape)
print("Number of test labels:", len(test_labels))

# normalize the images to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# function to display the images
def display_images(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

# display the first 25 images from the training set
display_images(train_images, train_labels)
