from datetime import datetime


def isp(i):
    i = str(i)
    return i == ''.join(reversed(i))

def enums():
    for i in range(1000000000):
        if isp(i) and isp(i*i):
            print i, i*i

print datetime.now()
ans = []
for i in xrange(1<<25):
    cnt = 0
    for c in str(i):
        if c == '1':
            cnt += 1
    if cnt < 10:
        ans.append(cnt)

print len(ans)
print datetime.now()