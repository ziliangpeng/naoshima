
def read_array(convertor=None):
    ret = raw_input().split()
    if convertor: ret = map(convertor, ret)
    return ret


def main():
    n, m = read_array(int)
    ns = read_array(int)
    ans = [0] * n
    s = set()
    for i in xrange(n-1, -1, -1):
        s.add(ns[i])
        ans[i] = len(s)

    for _ in xrange(m):
        l = input()
        print ans[l-1]


main()
