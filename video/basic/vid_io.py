import sys
import random
import cv2
import numpy
import math

filename = sys.argv[1]

vid = cv2.VideoCapture(filename)
width = int(vid.get(3))
height = int(vid.get(4))
print 'width = %s' % width
print 'height = %s' % height
fourcc = cv2.cv.CV_FOURCC(*'FMP4') # describe codec
out = cv2.VideoWriter('%s-output-%f.mp4' % (filename, random.random()), fourcc, 20, (width, height))

def img_diff(frame_cnt, prev, curr):
    if prev is None:
        return curr

    ret = curr.copy()
    for i in xrange(len(curr)):
        for j in xrange(len(curr[0])):
            ret[i][j][0] = numpy.uint8(max(0, int(curr[i][j][0]) - int(prev[i][j][0])))
            ret[i][j][1] = numpy.uint8(max(0, int(curr[i][j][1]) - int(prev[i][j][1])))
            ret[i][j][2] = numpy.uint8(max(0, int(curr[i][j][2]) - int(prev[i][j][2])))
    return ret

prev_img = None
frame_cnt = 0
while True:
    print 'frame_cnt %d' % frame_cnt
    success, img = vid.read()
    if success == False:
        break
    diff_img = img_diff(frame_cnt, prev_img, img)
    out.write(diff_img)

    prev_img = img
    frame_cnt += 1

vid.release()
out.release()
cv2.destroyAllWindows()
