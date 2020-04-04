
T = input()

for t in range(1, 1+T):
    N = input()
    tasks = []
    assigned = [' '] * N
    # print(assigned)
    for n in range(N):
        s, e = map(int, raw_input().split())
        tasks.append((n, s, e))
    tasks.sort(key=lambda task:task[1])
    cend = 0
    jend = 0
    impossible = False
    for task in tasks:
        n, s, e = task
        if cend <= s:
            assigned[n] = 'C'
            cend = e
        elif jend <= s:
            assigned[n] = 'J'
            jend = e
        else:
            impossible = True
            break

    if impossible:
        ans = "IMPOSSIBLE"
    else:
        ans = ''.join(assigned)

    print("Case #%d: %s" % (t, ans))
