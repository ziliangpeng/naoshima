from datetime import datetime
from collections import defaultdict
from hashlib import md5
import base64
import os


class FileGroup:

    def __init__(self, file_size, v):
        self.v = v
        self.file_size = file_size

    def __str__(self):
        ret = 'found duplicate files(size %d): \n' % (self.file_size)
        for f in self.v:
            ret += '  -- ' + f + '\n'
        return ret


class FullContentSanmpler:

    def __init__(self):
        pass

    def sample(self, filename):
        with open(filename, 'rb') as f:
            return f.read()


class HeadFixLengthContentSampler:

    def __init__(self, head_bytes=1024*1024):
        self.head_bytes = head_bytes

    def sample(self, filename):
        with open(filename, 'rb') as f:
            return f.read(self.head_bytes)


def detect(path, sampler=FullContentSanmpler(), verbose=False, fuzzy=False):

    result_dict = defaultdict(list)

    def process_file(path):
        if verbose:
            print 'examining:', path
        sz = os.path.getsize(path)
        b = sampler.sample(path)
        m = md5()
        m.update(b)
        hash_value = base64.b16encode(m.digest())

        file_size = os.stat(path).st_size

        result_dict[(hash_value, file_size)].append(path)

    def dfs(path):
        for sub_path in os.listdir(path):
            next_path = os.path.join(path, sub_path)
            if os.path.islink(next_path):
                continue
            elif os.path.isfile(next_path):
                try:
                    process_file(next_path)
                except Exception as e:
                    print e
                    print 'error processing file', next_path
            elif os.path.isdir(next_path):
                try:
                    dfs(next_path)
                except Exception as e:
                    print e
                    print 'error processing directory', next_path

    dfs(path)
    duplicate_files = []
    for k, v in result_dict.items():
        if len(v) > 1:
            duplicate_files.append(FileGroup(k[1], v))

    return duplicate_files


def detect_fuzzy(*kwargs):
    return detect(*kwargs, fuzzy=True)


if __name__ == '__main__':
    start_time = datetime.now()

    ret = detect_fuzzy('.')
    print ''
    for item in ret:
        print item

    end_time = datetime.now()
    print 'time spent:', end_time - start_time
