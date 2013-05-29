from datetime import datetime
from collections import defaultdict
from hashlib import md5
import base64
import os


class FileGroup:

    def __init__(self, v):
        self.v = v

    def __str__(self):
        ret = 'found duplicate files:\n'
        for f in self.v:
            ret += '  -- ' + f + '\n'
        return ret


def full_content_sampler(filename):
    with open(filename, 'rb') as f:
        return f.read()


def detect(path, sampler=None, verbose=False, fuzzy=False):

    result_dict = defaultdict(list)

    def process_file(path):
        if verbose:
            print 'examining:', path
        sz = os.path.getsize(path)
        b = full_content_sampler(path)
        m = md5()
        m.update(b)
        hash_value = base64.b16encode(m.digest())

        result_dict[hash_value].append(path)

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
            duplicate_files.append(FileGroup(v))

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
