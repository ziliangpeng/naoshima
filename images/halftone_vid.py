from halftone import Halftone
import sys
import os
import cv2
from PIL import Image, ImageDraw, ImageStat
import numpy

path = sys.argv[1]


def animate(file, vid, del1, del2, del3, del4, steps):
    for i in range(steps):
        h = Halftone(file)

        sample=10
        scale=1
        percentage=0
        # filename_addition="_halftoned"
        # angles=[0, 1, 3, 4]
        angles=[del1 * i, del2 * i, del3 * i, del4 * i]
        style="color"
        antialias=False

        f, e = os.path.splitext(path)

        try:
            im = Image.open(path)
        except IOError:
            raise

        if style == "grayscale":
            angles = angles[:1]
            gray_im = im.convert("L")
            dots = h.halftone(im, gray_im, sample, scale, angles, antialias)
            new = dots[0]
        else:
            cmyk = h.gcr(im, percentage)
            dots = h.halftone(im, cmyk, sample, scale, angles, antialias)
            new = Image.merge("CMYK", dots)

        imtemp = new.copy().convert('RGB')
        vid.write(cv2.cvtColor(numpy.array(imtemp), cv2.COLOR_RGB2BGR))

frame_width = 600 #new.width
frame_height = 600 # new.height
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('outpy.mkv',fourcc, 10, (frame_width,frame_height))

# animate(path, out, 1, 0, 0, 0, 90)
# animate(path, out, 0, 1, 0, 0, 90)
# animate(path, out, 0, 0, 1, 0, 90)
# animate(path, out, 0, 0, 0, 1, 90)

animate(path, out, 1, 2, 3, 4, 90)

out.release()

cv2.destroyAllWindows() 
