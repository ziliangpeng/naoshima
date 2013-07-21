from datetime import datetime
from collections import defaultdict
from hashlib import md5
import base64
import os
import string
import operator
from util import sort_files_by_create_time, get_creation_time


_debug = False


class FileType:
    Photo = ['jpg', 'jpeg', 'png', 'gif']
    Video = ['avi', 'mov', 'mp4', 'mpg']
    Document = ['pdf', 'doc', 'docx', 'pages']


class FileGroup:

    def __init__(self, file_size, v):
        self.v = v
        self._sort()
        self.file_size = file_size

    def _sort(self):
        sort_files_by_create_time(self.v)


    def append(self, val):
        self.v.append(val)
        self._sort()

    def count(self):
        return len(self.v)

    def __str__(self):
        ret = 'found duplicate files(size %d): \n' % (self.file_size)
        for f in self.v:
            ret += '  -- %s, %s\n' % (str(f), datetime.fromtimestamp(get_creation_time(f)))
        return ret
    
    def __getitem__(self, i):
        return self.v[i]


class FullContentSampler:

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


def match_file_type(fname, file_types):
    if file_types == None:
        return True
    else:
        return fname[fname.rfind('.'):][1:].lower() in file_types


def small_file_exact_match(file_group):
    file_contents = []
    new_file_groups = []

    for fname in file_group.v:
        with open(fname, 'rb') as f:
            content = f.read()
        for exist_content, new_file_group in zip(file_contents, new_file_groups):
            if exist_content == content:
                new_file_group.append(fname)
                break
        else:
            file_contents.append(content)
            new_file_groups.append(FileGroup(file_group.file_size, [fname]))

    return new_file_groups


""" Match algorithm for large files. This uses buffer for content comparasion 
    to avoid huge memory consumption.
"""
def big_file_match(fn1, fn2):
    with open(fn1, 'rb') as f1:
        with open(fn2, 'rb') as f2:
            buffer_size = 1024 * 1024
            while True:
                block1 = f1.read(buffer_size)
                block2 = f2.read(buffer_size)
                if block1 != block2:
                    return False
                if block1 == block2 == '':
                    return True


""" Converts fuzzy match group to exact match group."""
def big_file_exact_match(file_group):
    new_file_groups = []

    for fname in file_group.v:
        for new_file_group in new_file_groups:
            if big_file_match(new_file_group[0], fname):
                new_file_group.append(fname)
                break
        else:
            new_file_groups.append(FileGroup(file_group.file_size, [fname]))

    return new_file_groups


def exact_match(file_group):
    mb = 1024 * 1024
    big_file_threshold = _debug and mb or 100 * mb

    if file_group.file_size < big_file_threshold:
        return small_file_exact_match(file_group)
    else:
        return big_file_exact_match(file_group)


def detect(path, file_types=None, sampler=FullContentSampler(), verbose=False, fuzzy=False):
    
    if file_types != None:
        if isinstance(file_types, (str, unicode)):
            file_types = [file_types]

        file_types = set(map(string.lower, file_types))

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
                if match_file_type(sub_path, file_types):
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

    if not fuzzy:
        duplicate_files = reduce(operator.add, map(exact_match, duplicate_files), [])

    duplicate_files = sorted(duplicate_files, key=lambda x: x.file_size)

    return duplicate_files


def detect_fuzzy(*args, **kwargs):
    kwfuzzy = 'fuzzy'
    if kwfuzzy in kwargs:
        if kwargs[kwfuzzy] == True:
            del kwargs[kwfuzzy]
            return detect(*args, fuzzy=True, **kwargs)
        elif kwfuzzy == False:
            print 'keyword %s should be True' % (kwfuzzy)
            return
        else:
            print 'keyword %s must a boolean value' % (kwfuzzy)
            return



if __name__ == '__main__':
    start_time = datetime.now()

    ret = detect_fuzzy('.')
    print ''
    for item in ret:
        print item

    end_time = datetime.now()
    print 'time spent:', end_time - start_time
