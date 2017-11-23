import random
import math
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw

H = 512
W = 512
STAR_COUNTS = 50

im = Image.new('RGB', (H, W))

def rand_color():
    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        avg = (r + g + b) / 3
        if avg >= 128:
            return (r, g, b)


sheena_ringos = []
for _ in range(STAR_COUNTS):
    i = random.randint(0, H - 1)
    j = random.randint(0, W - 1)
    c = rand_color()
    sheena_ringos.append(((i, j), c))


def dist(i1, j1, i2, j2):
    return math.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)


"""
    Given a list of distances (floating points), returns a list of proportions each dist takes.
    - smaller distance should get larger proportion
    - dist 0 should get 1.0
    - sum of all proportion should be 1
"""
def cal_proportions(dists):
    sum_dist = sum(dists)
    reverses = [(sum_dist / d) ** 1.5 for d in dists]
    sum_reverse = sum(reverses)
    proportions = [r / sum_reverse for r in reverses]
    return proportions


def cal(i, j):
    def sort_sr(a, b):
        dist_a = dist(a[0][0], a[0][1], i, j)
        dist_b = dist(b[0][0], b[0][1], i, j)
        return dist_a - dist_b

    sr = [(dist(s[0][0], s[0][1], i, j), s[1]) for s in sheena_ringos]
    sr.sort()
    if sr[0][0] == 0.0:
        return sr[0][1]
    proportions = cal_proportions([s[0] for s in sr])
    r, g, b = 0, 0, 0
    for s, p in zip(sr, proportions):
        sc = s[1]
        sr, sg, sb = sc
        r += sr * p
        g += sg * p
        b += sb * p
    return (int(r), int(g), int(b))


for i in range(H):
    for j in range(H):
        # print(i, j)
        c = cal(i, j)
        im.putpixel((i, j), c)

im.show()
