from PIL import Image, ImageMode
import glog
from scipy.spatial import distance

CIRCLE_DIAMETER = 256 * 3


MAX_COLOR = (255, 255, 255)

if __name__ == '__main__':
    glog.info("Diameter is %d" % (CIRCLE_DIAMETER))
    height, width = CIRCLE_DIAMETER, CIRCLE_DIAMETER
    img = Image.new('RGB', (width, height))

    RADIUS = CIRCLE_DIAMETER / 2
    center = (RADIUS, RADIUS)
    glog.info("circle center is %s" % (str(center)))


    px = img.load()
    for i in range(width):
        for j in range(height):
            dist = distance.euclidean((i, j), center)
            gradient = max(0, RADIUS - dist) / RADIUS
            px[i, j] = (int(gradient * MAX_COLOR[0]), int(gradient * MAX_COLOR[1]), int(gradient * MAX_COLOR[2]))

    img.save("output.jpg")