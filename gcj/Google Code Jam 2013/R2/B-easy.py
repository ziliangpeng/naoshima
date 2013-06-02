from collections import defaultdict

def read_array(convertor=None):
    ret = raw_input().split()
    if convertor: ret = map(convertor, ret)
    return ret


def count_bit(x):
    bt = 0
    while x:
        if x % 2 == 1:
            bt += 1
        x /= 2
    return bt


def first_zero(x, N):
    for i in range(N-1, 0, -1):
        if 2 ** i <= N:
            N -= 2 ** i
        else:
            return i


""" could """
def cal_ans1(P, N):
    m = defaultdict(list)

    for i in range(2 ** N):
        bc = count_bit(i)
        m[bc].append(i)

    for v in m.values():
        v.sort()

    assigned = []
    for i in range(P):
        bc = count_bit(i)
        v = m[bc]
        val = v[-1]
        assigned.append(val)
        v.remove(val)

    return max(assigned)


    """
    btc = count_bit(P-1)
    lose = btc

    return 2 ** N - btc


    pass"""


""" guarantee"""
def cal_ans2(P, N):
    m = defaultdict(list)

    for i in range(2 ** N):
        bc = count_bit(i)
        m[bc].append(i)

    for v in m.values():
        v.sort()

    assigned = []
    for i in range(P):
        bc = count_bit(i)
        v = m[bc]
        val = v[-1]
        assigned.append(val)
        v.remove(val)

    assigned.sort()

    for i in range(len(assigned)):
        if assigned[i] != i:
            return i - 1

    return assigned[-1]

"""
    pos = first_zero(P-1, N)
    left_c = N - 1 - pos

    return left_c
    """


def main():
    T = input()
    for _ in range(T):
        N, P = read_array(int)
        _P = P

        P = _P
        ans1 = cal_ans1(P, N)
        ans2 = cal_ans2(P, N)

        print 'Case #%d: %d %d' % (_+1, ans2, ans1)



main()
