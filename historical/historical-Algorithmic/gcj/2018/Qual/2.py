
T = int(input())
for t in range(1, T+1):
    N = int(input())
    values = list(map(int, input().split()))
    v1 = [values[i] for i in range(0, N, 2)]
    v2 = [values[i] for i in range(1, N, 2)]
    v1.sort()
    v2.sort()
    v = [0] * N
    for i, x in enumerate(v1):
        v[i * 2] = x
    for i, x in enumerate(v2):
        v[i * 2 + 1] = x
    sv = list(sorted(v))
    # print(v)
    # print(sv)
    ans = -1
    for i in range(len(v)):
        if v[i] != sv[i]:
            ans = i
            break

    print("Case #%d: %s" % (t, ans == -1 and "OK" or str(ans)))

