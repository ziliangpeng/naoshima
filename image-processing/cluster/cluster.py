from PIL import Image, ImageMode
import glog
from random import randint
import math


class RGB():
    @classmethod
    def random(cls):
        def r(): return randint(0, 255)
        return RGB(r(), r(), r())

    @classmethod
    def avg(cls, pixels):
        cnt = len(pixels)
        if cnt == 0:
            glog.warn("Result may be invalid.")
            return RGB(0, 0, 0)
        r = sum([p.r for p in pixels])
        g = sum([p.g for p in pixels])
        b = sum([p.b for p in pixels])
        return RGB(r // cnt, g // cnt, b // cnt)

    def to_tuple(self):
        return (self.r, self.g, self.b)

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __eq__(self, o):
        return self.r == o.r and self.g == o.g and self.b == o.b

    def __str__(self):
        return "RGBpixel(%d, %d, %d)" % (self.r, self.g, self.b)

    def __repr__(self):
        return "RGB(%d, %d, %d)" % (self.r, self.g, self.b)

    def distance(self, other_pixel):
        return abs(self.r - other_pixel.r) + abs(self.g - other_pixel.g) + abs(self.b - other_pixel.b)


def kmeans(pixels, k):

    # def find_closest(pixel, means):

    means = [RGB.random() for i in range(k)]
    glog.debug("Initial mean colors are " + str(means))

    while True:
        belonging = [[] for i in range(k)]
        for pixel in pixels:
            distances = [(pixel.distance(means[i]), i) for i in range(k)]
            min_distance = min(distances)
            index = min_distance[1]
            belonging[index].append(pixel)

        new_means = [RGB.avg(belonging[i]) for i in range(k)]
        glog.debug("New mean colors are " + str(new_means))
        if new_means == means:
            glog.debug("New mean has not changed. End iteration.")
            break
        else:
            means = new_means

    return means


def cluster(img, k) -> Image:
    height = img.height
    width = img.width
    glog.info("Processing image with height(%d) and width(%d)" %
              (height, width))

    pixels = []
    for j in range(height):
        for i in range(width):
            coordinate = (i, j)
            pixel = img.getpixel(coordinate)
            rgb_pixel = RGB(pixel[0], pixel[1], pixel[2])
            pixels.append(rgb_pixel)

    means = kmeans(pixels, k)
    glog.debug("Final mean colors are " + str(means))

    return means


if __name__ == '__main__':
    glog.setLevel('DEBUG')
    K = 3
    img = Image.open("test.jpg")
    colors = cluster(img, K)

    # Vertical strip
    strip_multiplier = 10
    strip_height = 100 * strip_multiplier
    strip_width = 50 * strip_multiplier
    new_img = Image.new('RGB', (strip_width * K, strip_height))
    px = new_img.load()
    for j in range(strip_height):
        for i in range(strip_width * K):
            px[i, j] = colors[i // strip_width].to_tuple()

    new_img.save("test.jps.cluster.jpg")
