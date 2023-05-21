#!/usr/bin/python3

import ffmpeg as ff
from os import utime
from pprint import pprint
from datetime import datetime
import calendar
import sys

filename = 'DJI_0550_001.MP4-converted-br-33177600.mp4'
filename = sys.argv[1]

stream = ff.input(filename)

cdate_str = ''

for s in ff.probe(filename)["streams"]:
  if s['codec_type'] == 'video':
    cdate_str = s['tags']['creation_time']
assert cdate_str

print(cdate_str)

d = datetime.strptime(cdate_str.split('.')[0], '%Y-%m-%dT%H:%M:%S')
print(d)
epoch = calendar.timegm(d.timetuple())
print(epoch)

utime(filename, (epoch, epoch))

