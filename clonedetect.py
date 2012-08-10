import os
import md5, base64

d = {}

def process_file(path):
    print 'eximining ', path
    sz = os.path.getsize(path)
    f = open(path)
    b = f.read(200 * 1024)
    f.close()
    m = md5.new()
    m.update(b)
    md5_value = base64.b16encode(m.digest())

    if md5_value not in d: d[md5_value] = []
    d[md5_value].append((path, sz))
    

def dfs(path):
    for sub_path in os.listdir(path):
        next_path = os.path.join(path, sub_path)
        if os.path.isfile(next_path):
            try:
                process_file(next_path)
            except:
                print 'error processing file', next_path
        elif os.path.isdir(next_path):
            try:
                dfs(next_path)
            except:
                print 'error processing directory', next_path


cwd = os.getcwd()

dfs(cwd)

for k, v in d.iteritems():
    if len(v) > 1:
        print 'found ========================='
        for p in v:
            print p
