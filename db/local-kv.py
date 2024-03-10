import time
import random
import os
import statsd

import lsm
import shelve
import dbm
import os

"""
tries to repeatedly generate keys within a relatively small range.
dbm uses B-tree and size stops growing.
lsm will continue to grow.

bechmark shows the app is cpu-bound. so we cannot tell write throughtput difference between dbm and lsm.
"""


s = statsd.StatsClient("localhost", 8125)
# sd = datadog.statsd


if os.path.isdir('/tmp/db'):
    for filename in os.listdir('/tmp/db'):
        file_path = os.path.join('/tmp/db', filename)
        os.remove(file_path)
else:
    os.makedirs('/tmp/db')

lsmdb = lsm.LSM('/tmp/db/lsmdb')
lsmdb.open()

# sl = shelve.open('/tmp/db/shelve', 'c')
# sl_gnu = shelve.open('/tmp/db/shelve_gnu', dbm='dbm.gnu')
# sl_ndbm = shelve.open('/tmp/db/shelve_ndbm', dbm='dbm.ndbm')
# sl_dumb = shelve.open('/tmp/db/shelve_dumb', dbm='dbm.dumb')

# db = dbm.open('/tmp/db/dbm', 'c')

def rand_int(upper=1000000000):
    i = random.randint(1, upper)
    return i

def rand_str(length=100):
    s = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))
    return s

for i in range(1000000000):
    key_range = 10000000
    start_time = time.time()
    k = str(rand_int(key_range))
    k_enc = k.encode('utf-8')
    v = rand_str(rand_int(1000))
    v_enc = v.encode('utf-8')
    end_time = time.time()
    elapsed = int((end_time - start_time) * 1000000000)
    s.gauge('lsmdb.keycreate', elapsed)

    start_time = time.time()
    lsmdb[k_enc] = v_enc
    end_time = time.time()
    elapsed = int((end_time - start_time) * 1000000000)
    s.gauge('lsmdb.insert', elapsed)
    s.incr('lsmdb.cnt', 1)

    k = str(rand_int(key_range))
    k_enc = k.encode('utf-8')
    start_time = time.time()
    key_exist = k_enc in lsmdb
    end_time = time.time()
    elapsed = int((end_time - start_time) * 1000000000)
    s.gauge('lsmdb.exist', elapsed)
    if key_exist:
        start_time = time.time()
        result = lsmdb[k_enc]
        end_time = time.time()
        elapsed = int((end_time - start_time) * 1000000000)
        s.gauge('lsmdb.get', elapsed)
        s.incr('lsmdb.hit', 1)
    else:
        s.incr('lsmdb.miss', 1)

    if i % 100000 == 0:
        file_size = os.path.getsize('/tmp/db/lsmdb')
        s.gauge('lsmdb.filesize', file_size)

    # sl[k] = v
    # db[k] = v
    # sl_gnu[k] = v
    # sl_ndbm[k] = v
    # sl_dumb[k] = v
    # print(f"insert time {elapsed} ns")
    # lsmdb.flush()
    # print(len(lsmdb))
    # file_path = '/tmp/lsmdb'
    # file_size = os.path.getsize(file_path)
    # print(f"File size: {file_size} bytes")
# input("Press Enter to continue...")

with lsm.LSM('/tmp/db/lsmdb') as source_lsm:
    with lsm.LSM('/tmp/db/lsmdb.copy') as target_lsm:
        for key, value in source_lsm.items():
            target_lsm[key] = value