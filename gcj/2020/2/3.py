from collections import defaultdict


def solve(pts):
    graphs = defaultdict(list)
    for p in pts:
        x, y = p
        graphs[x].append(y)
    odds = 0
    ones = 0
    totals = 0
    for k in graphs:
        v = len(graphs[k])
        if v == 1:
            ones += 1
        elif v % 2 == 0:
            totals += v
        else:
            totals += v / 2 * 2
            odds += 1

        # if v % 2 == 0:
        #     evens += 1
        # else:
        #     odds += 1
    return totals + min(2, ones) + odds / 2 * 2


T = input()
for t in range(1, T + 1):
    N = input()
    pts = []
    pts2 = []
    for i in range(N):
        x, y = map(int, raw_input().split())
        pts.append((x, y))
        pts2.append((y, x))
    ans1 = solve(pts)
    ans2 = solve(pts2)
    ans = max(ans1, ans2)
    print("Case #%d: %s" % (t, ans))
