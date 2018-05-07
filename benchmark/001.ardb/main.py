import redis
import time
import datetime


host = 'localhost'
port = 6379
db = 1
r = redis.Redis(host, port, db)


start = time.time()
i = 0
ttl = 300
while time.time() - start < 256: # repeat for 1 minute
    i += 1
    key = 'k' + str(i)
    value = str(i)[0] * 16
    r.set(key, value, ttl)

print('done')
print(len(r.keys()))

