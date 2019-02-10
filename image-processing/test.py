# https://tonysyu.github.io/ipython-jupyter-widgets-an-image-convolution-demo.html#.XF-6oBlKh25


import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'


###

from skimage import data, filters, io

image = data.camera()
fname = 'chelmico.jpg'
image = io.imread(fname)
# Ignore the Gaussian filter, for now.
# (This is explained at the end of the article.)
smooth_image = filters.gaussian(image, 5)
# plt.imshow(smooth_image)
# plt.show()


###

import numpy as np

horizontal_edge_kernel = np.array([[ 1,  2,  1],
                                   [ 0,  0,  0],
                                   [-1, -2, -1]])
# Use non-gray colormap to display negative values as red and positive
# values as blue.
plt.imshow(horizontal_edge_kernel, cmap=plt.cm.RdBu)
plt.show()