from collections import defaultdict
from datetime import datetime
import os
import time
import cv2
import pyheif
import face_recognition
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS, GPSTAGS
import pillow_heif
from PIL import ExifTags
from pillow_heif import register_heif_opener
import numpy as np

WORKING_DIR = "ichi-video"
VIDEO_HEIGHT = 2048
VIDEO_WIDTH = 2048
FACE_WIDTH = 512


def find_face(fname):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(fname)

    # Find all faces in the image
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) != 1:
        print(f"should be only one face {fname} but got {len(face_locations)}")
        return None

    return face_locations[0]


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
    return None


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

def get_files():
    print(os.listdir(WORKING_DIR))
    files = os.listdir(WORKING_DIR)
    files = [os.path.join(WORKING_DIR, f) for f in files]  # add path to each file
    files = filter(os.path.isfile, files)
    files = filter(lambda x: not x.endswith(".DS_Store"), files)
    files = filter(lambda x: not x.endswith(".MP4"), files)
    files = filter(img_ctime_exif, files)
    files = [f for f in files]
    files.sort(key=img_ctime_exif)
    return files


def make_video(img_count=1000000000):
    files = get_files()

    now = datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    video = cv2.VideoWriter(f"ichi-{now_string}.avi", 0, 1, (VIDEO_HEIGHT, VIDEO_WIDTH))

    for filename in files[:img_count]:
        print(filename)
        print(img_ctime_exif(filename))
        # Open an image file with Pillow
        pil_image = Image.open(filename)
        pil_image = transform_face(filename, pil_image)
        if pil_image is None:
            continue

        d = ImageDraw.Draw(pil_image)

        # Choose a font
        # Calculate the width and height of the text
        days = get_days(filename)
        text = str(days)
        # font = ImageFont.load_default()
        font = ImageFont.truetype("Keyboard.ttf", 256)
        textbbox_val = d.textbbox((0,0), text, font=font)
        _, _, text_width, text_height = textbbox_val

        # print(textbbox_val)
        # text_width, text_height = d.textsize(text, font)
        # Calculate the x and y coordinates for the text
        x = (pil_image.width - text_width) / 2
        y = pil_image.height - text_height - 21
        # Add text to image at calculated coordinates, fill color and font
        d.text((x, y), text, fill=(255,255,255), font=font)


        # Convert the Pillow image into a numpy array
        numpy_image = np.array(pil_image)
        # Convert the numpy array to a color image that OpenCV can handle
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        crop_img = opencv_image[0:VIDEO_HEIGHT, 0:VIDEO_WIDTH]
        video.write(crop_img)

    cv2.destroyAllWindows()
    video.release()

def get_days(filename):
    birth = datetime(2022, 7, 17)
    ctime = img_ctime_exif(filename)
    cdate = ctime.split(" ")[0]
    y, m, d = cdate.split(":")

    cdate = datetime(int(y), int(m), int(d))
    difference = cdate - birth
    days = difference.days
    return days



def stats():
    birth = datetime(2022, 7, 17)

    files = get_files()
    week_dict = defaultdict(int)
    month_dict = defaultdict(int)

    with open("ichi_img_dates.txt", "w") as f:
        for file in files:
            if not find_face(file):
                continue
            ctime = img_ctime_exif(file)
            cdate = ctime.split(" ")[0]
            y, m, d = cdate.split(":")
            f.write(f"{y}-{m}-{d}\n")

            cdate = datetime(int(y), int(m), int(d))
            difference = cdate - birth
            days = difference.days

            weeks = days // 7
            month = f"{y}-{m}"
            week_dict[weeks] += 1
            month_dict[month] += 1

    with open("ichi_weeks.txt", "w") as f:
        for week, count in week_dict.items():
            f.write(f"{week} {count}\n")

    with open("ichi_months.txt", "w") as f:
        for month, count in month_dict.items():
            f.write(f"{month} {count}\n")



def main():
    make_video()
    # stats()


if __name__ == "__main__":
    register_heif_opener()
    main()
