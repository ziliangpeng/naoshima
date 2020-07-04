from halftone import Halftone
import sys
import os
import cv2
from PIL import Image, ImageDraw, ImageStat

path = sys.argv[1]

h = Halftone(path)

sample=10
scale=1
percentage=0
filename_addition="_halftoned"
angles=[0, 1, 3, 4]
style="color"
antialias=False

f, e = os.path.splitext(path)

outfile = "%s%s%s" % (f, filename_addition, e)

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

# new.save(outfile)
# vid = cv2.VideoCapture("out.mp4")
print(dir(new))
print(new.size)
frame_width = new.width
frame_height = new.height
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

