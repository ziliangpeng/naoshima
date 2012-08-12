import os
import hashlib
import base64


def sample_file_content(f, sz):
    return f.read(200 * 1024)


def process_file(path, result_dict):
    print 'examining:', path
    sz = os.path.getsize(path)
    f = open(path)
    b = sample_file_content(f, sz)
    f.close()
    m = hashlib.md5()
    m.update(b)
    md5_value = base64.b16encode(m.digest())

    if md5_value not in result_dict:
        result_dict[md5_value] = []
    result_dict[md5_value].append((path, sz))
    

def dfs(path, result_dict):
    for sub_path in os.listdir(path):
        next_path = os.path.join(path, sub_path)
        if os.path.islink(next_path):
            continue
        elif os.path.isfile(next_path):
            try:
                process_file(next_path, result_dict)
            except:
                print 'error processing file', next_path
        elif os.path.isdir(next_path):
            try:
                dfs(next_path, result_dict)
            except:
                print 'error processing directory', next_path


def main():
    cwd = os.getcwd()
    result_dict = {}
    dfs(cwd, result_dict)

    for k, v in result_dict.iteritems():
        if len(v) > 1:
            print 'found ========================='
            for p in v:
                print p


if __name__ == '__main__':
    main()