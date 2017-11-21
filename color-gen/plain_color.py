import random
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw

H = 1024
W = 1024

im = Image.new('RGB', (H, W))

while True:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    print(r, g, b)
    avg = (r + g + b) / 3
    print('avg', avg)
    if avg >= 128:
        break

c = (r, g, b)

# brute force the fill
# for i in range(H):
#     for j in range(W):
#         im.putpixel((i, j), c)

# faster
draw = ImageDraw.Draw(im)
draw.rectangle([(0, 0), (H, W)], c)

im.show()
