import math

T = int(input())


def cal(R, B, C, Ms, Ss, Ps, bt):
    pos = []
    bt2 = bt
    for bi in range(B):
        pos.append(bt2 % C)
        bt2 //= C

for t in range(1, T+1):
    R, B, C = list(map(int, input().split()))
    Ms, Ss, Ps = [], [], []
    for c in range(C):
        m, s, p = list(map(int, input().split()))
        Ms.append(m)
        Ss.append(s)
        Ps.append(p)

    best = 10 ** 10
    for bt in range():
        t = cal(R, B, C, Ms, Ss, Ps, bt)
