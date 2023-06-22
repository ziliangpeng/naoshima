import face_recognition
from PIL import Image, ImageDraw

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("test.jpg")

# Find all faces in the image
face_locations = face_recognition.face_locations(image)

# Convert the numpy array image into pil image object
pil_image = Image.fromarray(image)

# Create a ImageDraw instance
d = ImageDraw.Draw(pil_image)

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # Draw a box around faces
    d.rectangle([left, top, right, bottom], outline=(255, 255, 255))

# Show the picture
pil_image.show()
