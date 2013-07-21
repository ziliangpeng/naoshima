from datetime import datetime

from decl import detect_fuzzy, detect
from decl import HeadFixLengthContentSampler, FullContentSampler
from decl import FileType
import shutil
import os

def move_to(fname, trash_box='./trash_box_'):
    des = trash_box + fname[1:]
    des_dir = des[:des.rfind('/')]
    try:
        os.makedirs(des_dir)
    except Exception as e:
        print e

    print 'moving', fname, 'to', des_dir

    shutil.move(fname, des_dir)


def main():
    start_time = datetime.now()

    save_size = 0

    ret = detect('.', file_types=FileType.Video+FileType.Photo, \
            sampler=FullContentSampler(), verbose=False)
    print ''
    for item in ret:
        #print item
        vs = item.v

        for v in vs:
            print v
        print '--'

        for fname in vs[1:]:
            pass
            move_to(fname)

        save_size += item.file_size * (item.count() - 1)

    end_time = datetime.now()
    print 'time spent:', end_time - start_time
    print 'save bytes:', save_size


if __name__ == '__main__':
    main()
