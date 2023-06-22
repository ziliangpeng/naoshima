import os
import time
import cv2
import pyheif
import face_recognition
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS, GPSTAGS
import pillow_heif
from PIL import ExifTags
from pillow_heif import register_heif_opener
import numpy as np

VIDEO_HEIGHT = 2048
VIDEO_WIDTH = 2048
FACE_WIDTH = 512


# def heic_to_jpg(heic_path, jpg_path):
#     heif_file = pyheif.read(heic_path)
#     image = Image.frombytes(
#         heif_file.mode, 
#         heif_file.size, 
#         heif_file.data,
#         "raw",
#         heif_file.mode,
#         heif_file.stride,
#     )
#     return image

def find_face(fname):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(fname)

    # Find all faces in the image
    face_locations = face_recognition.face_locations(image)


    # Convert the numpy array image into pil image object
    pil_image = Image.fromarray(image)

    # Create a ImageDraw instance
    d = ImageDraw.Draw(pil_image)

    # assert len(face_locations) == 1, f"There should be only one face in the image {fname} but we got {len(face_locations)}"
    if len(face_locations) != 1:
        print(f"There should be only one face in the image {fname} but we got {len(face_locations)}")
        return None

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Draw a box around faces
        d.rectangle([left, top, right, bottom], outline=(255, 255, 255))

    # Show the picture
    # pil_image.show()
    return face_locations[0]


def heif_ctime(img_path):
    image = Image.open(img_path)
    image_exif = image.getexif()
    print(image_exif)

def transform_face(filename, image):
    face_loc = find_face(filename)
    if face_loc is None:
        return None
    top, right, bottom, left = face_loc

    # Calculate face width and image dimensions
    face_width = right - left
    image_width, image_height = image.size

    # Calculate the scale factor
    scale = FACE_WIDTH / face_width

    # Calculate new image dimensions
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    # Resize the image
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the center of the face in the resized image
    face_center_x = (left + face_width / 2) * scale
    face_center_y = (top + (bottom - top) / 2) * scale

    # Calculate crop coordinates
    left = int(face_center_x - VIDEO_WIDTH / 2)
    top = int(face_center_y - VIDEO_HEIGHT / 2)
    right = left + VIDEO_WIDTH
    bottom = top + VIDEO_HEIGHT

    # Crop the image
    image = image.crop((left, top, right, bottom))
    # image.show()

    return image


def make_video():

    def img_ctime_exif(img_path):
        image = Image.open(img_path)
        exif_data = image.getexif()
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                
                if tag_name == "DateTime":
                    # print(f"Image creation time: {value}")
                    return value
        print("No exif data found for ", img_path)
        print(exif_data)

    # list all file and sort by file creation date:
    working_dir = "ichi-video"
    print(os.listdir(working_dir))
    files = os.listdir(working_dir)
    # files = filter(lambda x: not x.endswith(".HEIC"), files)
    files = filter(lambda x: not x.endswith(".DS_Store"), files)
    files = [os.path.join(working_dir, f) for f in files] # add path to each file
    files = filter(img_ctime_exif, files)
    files = filter(os.path.isfile, files)
    files = [f for f in files]
    files.sort(key=img_ctime_exif) # os.path.getctime(x))
    # print("sorted")
    for file in files:
        # mtime = os.path.getctime(file)
        ctime = img_ctime_exif(file)
        # print(ctime)
        # convert mtime to string
        # str_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ctime))
        str_time = ctime

        # print(file, str_time)

    # video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, (1024, 1024))
    video = cv2.VideoWriter('video.avi', 0, 1, (VIDEO_HEIGHT, VIDEO_WIDTH))


    # loop thorugh all files in the directory
    for filename in files:
        # if filename.endswith(".HEIC"):
        #     continue
        if filename.endswith('.DS_Store'):
            continue

        print(filename)
        full_fulename = filename
        # heic_to_jpg(full_fulename, "output.jpg")
        # location = show_face(full_fulename)

        # for image in processed_images:
            # video.write(cv2.imread(image))

        # Open an image file with Pillow
        pil_image = Image.open(full_fulename)
        pil_image = transform_face(full_fulename, pil_image)
        if pil_image is None:
            continue
        # Convert the Pillow image into a numpy array
        numpy_image = np.array(pil_image)
        # Convert the numpy array to a color image that OpenCV can handle
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img = opencv_image

        # img = cv2.imread(full_fulename)
        crop_img = img[0:VIDEO_HEIGHT, 0:VIDEO_WIDTH]
        # video.write(cv2.imread(full_fulename))
        video.write(crop_img)

    cv2.destroyAllWindows()
    video.release()


register_heif_opener()

# heif_ctime("ichi-video/IMG_5764.HEIC")

make_video()