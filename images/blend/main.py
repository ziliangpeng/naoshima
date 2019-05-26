from PIL import Image, ImageMode
import gflags
import sys

gflags.DEFINE_string('f1', '1.jpg', 'file 1')
gflags.DEFINE_string('f2', '2.jpg', 'file 2')
gflags.DEFINE_string('out', 'out.jpg', 'out file')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

file_1 = FLAGS.f1
file_2 = FLAGS.f2

img_1 = Image.open(file_1)
img_2 = Image.open(file_2)
assert(img_1.width == img_2.width)
assert(img_1.height == img_2.height)
w, h = img_1.width, img_1.height

px_1 = img_1.load()
px_2 = img_2.load()

img_out = Image.new('RGB', (w, h))
px_out = img_out.load()
for i in range(w):
    for j in range(h):
        rgb_1 = px_1[i, j] 
        rgb_2 = px_2[i, j] 
        px_out[i, j] = (int(rgb_1[0]/2 + rgb_2[0]/2), int(rgb_1[1]/2 + rgb_2[1]/2), int(rgb_1[2]/2 + rgb_2[2]/2))

img_out.save(FLAGS.out)