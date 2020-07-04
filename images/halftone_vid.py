from halftone import Halftone
import sys
import os
import cv2
from PIL import Image, ImageDraw, ImageStat
import numpy
import datetime
import glog

path = sys.argv[1]


def animate(im, vid, del1, del2, del3, del4, steps):
    for i in range(steps):
        glog.info('Iter %d' % (i))
        h = Halftone("")

        sample = 10
        scale = 1
        percentage = 0
        angles = [del1 * i, del2 * i, del3 * i, del4 * i]
        style = "color"
        antialias = False

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


f, e = os.path.splitext(path)

try:
    im = Image.open(path)
except IOError:
    raise

frame_width = im.width
frame_height = im.height
glog.info('size: %d, %d' % (frame_width, frame_height))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'out_%d.mp4' % (int(datetime.datetime.utcnow().timestamp())), fourcc, 10,
    (frame_width, frame_height))

# animate(path, out, 1, 0, 0, 0, 90)
# animate(path, out, 0, 1, 0, 0, 90)
# animate(path, out, 0, 0, 1, 0, 90)
# animate(path, out, 0, 0, 0, 1, 90)

animate(im, out, 1, 2, 3, 4, 90)

out.release()

cv2.destroyAllWindows()
