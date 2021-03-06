from PIL import Image, ImageMode
import glog
from random import randint
import math
import sys
import math
import gflags

gflags.DEFINE_integer('k', 5, 'Number of clusters')
gflags.DEFINE_string('f', 'test.jpg', 'Filename')
gflags.DEFINE_string('comment', 'default', 'Comment')
# modes:
#   - colors
#   - img_with_colors
#   - img_with_colors_progress
gflags.DEFINE_string('mode', 'img_with_colors', 'Attach strip to image')


class RGB():
    @classmethod
    def random(cls):
        def r(): return randint(0, 255)
        return RGB(r(), r(), r())

    # TODO: move out of RGB
    @classmethod
    def fixed_seeds(cls, k):
        BLACK = RGB(0, 0, 0)

        RED = RGB(255, 0, 0)
        GREEN = RGB(0, 255, 0)
        BLUE = RGB(0, 0, 255)

        YELLOW = RGB(255, 255, 0)
        CYAN = RGB(0, 255, 255)
        MAGENTA = RGB(255, 0, 255)

        WHITE = RGB(255, 255, 255)

        if k == 2:
            return [WHITE, BLACK]
        if k == 3:
            return [RED, GREEN, BLUE]
        if k == 4:
            return [WHITE, RED, GREEN, BLUE]
        if k == 5:
            return [WHITE, RED, GREEN, BLUE, BLACK]
        if k == 6:
            return [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA]
        if k == 7:
            return [WHITE, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA]
        if k == 8:
            return [WHITE, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, BLACK]
        return [WHITE, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, BLACK] + [cls.random() for i in range(k - 8)]

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

    def __lt__(self, o):
        if self.r != o.r:
            return self.r < o.r
        if self.g != o.g:
            return self.g < o.g
        if self.b != o.b:
            return self.b < o.b
        return False

    def distance(self, other_pixel):
        return math.sqrt((self.r - other_pixel.r) ** 2 + (self.g - other_pixel.g) ** 2 + (self.b - other_pixel.b) ** 2)


def kmeans(pixels, k):

    # def find_closest(pixel, means):

    # means = [RGB.random() for i in range(k)]
    means = RGB.fixed_seeds(k)
    all_means = [means]
    glog.debug("Initial mean colors are " + str(means))

    while True:
        belonging = [[] for i in range(k)]
        for pixel in pixels:
            distances = [(pixel.distance(means[i]), i) for i in range(k)]
            min_distance = min(distances)
            index = min_distance[1]
            belonging[index].append(pixel)

        new_means = [RGB.avg(belonging[i]) for i in range(k)]
        all_means.append(new_means)
        glog.debug("New mean colors are " + str(new_means))
        diff = sum([means[i].distance(new_means[i]) for i in range(k)])
        means = new_means

        if diff < k * 5:
            glog.debug("New mean (almost) has not changed. End iteration.")
            break

    return all_means


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

    all_means = kmeans(pixels, k)
    means = all_means[-1]
    glog.debug("Final mean colors are " + str(means))

    return all_means


if __name__ == '__main__':
    glog.setLevel('DEBUG')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    filename = FLAGS.f
    K = FLAGS.k
    comment = FLAGS.comment

    strip_multiplier = 10
    mode = FLAGS.mode

    if mode == 'img_with_colors':
        img = Image.open(filename)
        all_means = cluster(img, K)
        colors = all_means[-1]
        colors.sort()

        strip_height = img.height // 3
        img_width = img.width
        img_height = img.height + strip_height
        new_img = Image.new('RGB', (img_width, img_height))

        # Copy original image
        input_px = img.load()
        output_px = new_img.load()
        for j in range(img.height):
            for i in range(img.width):
                output_px[i, j] = input_px[i, j]

        # Draw strip
        strip_start_j = img.height
        for j in range(strip_height):
            for i in range(img_width):
                color_i = min(K - 1, int(i // (img_width / K)))
                output_px[i, strip_start_j + j] = colors[color_i].to_tuple()

        new_img.save("%s.cluster.%d%s.inplace.jpg" %
                     (filename, K, '.' + comment if comment else ''))
    elif mode == 'colors':
        img = Image.open(filename)
        all_means = cluster(img, K)
        colors = all_means[-1]
        colors.sort()

        # Vertical strip
        strip_height = 100 * strip_multiplier
        strip_width = 50 * strip_multiplier
        new_img = Image.new('RGB', (strip_width * K, strip_height))
        px = new_img.load()
        for j in range(strip_height):
            for i in range(strip_width * K):
                px[i, j] = colors[i // strip_width].to_tuple()
        new_img.save("%s.cluster.%d.%s.jpg" %
                     (filename, K, '.' + comment if comment else ''))

    elif mode == 'img_with_colors_progress':
        img = Image.open(filename)
        all_means = cluster(img, K)

        strip_height = img.height // 10
        img_width = img.width
        img_height = img.height + strip_height * len(all_means)
        new_img = Image.new('RGB', (img_width, img_height))

        # Copy original image
        input_px = img.load()
        output_px = new_img.load()
        for j in range(img.height):
            for i in range(img.width):
                output_px[i, j] = input_px[i, j]

        # Draw strip
        for i, colors in enumerate(all_means):
            colors.sort()
            strip_start_j = img.height + strip_height * i
            for j in range(strip_height):
                for i in range(img_width):
                    color_i = min(K - 1, int(i // (img_width / K)))
                    output_px[i, strip_start_j +
                              j] = colors[color_i].to_tuple()

        new_img.save("%s.cluster.%d%s.progress.jpg" %
                     (filename, K, '.' + comment if comment else ''))
    else:
        glog.fatal("mode %s is not supported." % (mode))
        sys.exit(1)
