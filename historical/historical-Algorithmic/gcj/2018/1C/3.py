import sys

sys.setrecursionlimit(110000)
MAX_W = 10 ** 12

# the min total weight stack starting with 'i' and 'h' height
def read_cal(N, ws, i, h, cache, max_h):
    if i >= N:
        return MAX_W
    if h > max_h[i]:
        return MAX_W

    if (i, h) in cache:
        return cache[(i, h)]

    if h == 1:
        this_min_w = ws[i]
    else:
        pre_min_w = read_cal(N, ws, i+1, h-1, cache, max_h)
        if pre_min_w <= ws[i] * 6:
            this_min_w = pre_min_w + ws[i]
        else:
            this_min_w = MAX_W

    pre_min_w = read_cal(N, ws, i+1, h, cache, max_h)
    ans = min(this_min_w, pre_min_w)
    cache[(i, h)] = ans
    return ans


def solve_easy(N, weights):
    cache = {}
    max_h = [MAX_W for _ in range(N)]
    best_ans = 0
    for last_ant in range(N):
        for height in range(1, N+1):
            min_weight = read_cal(N, weights, last_ant, height, cache, max_h)
            if min_weight != MAX_W:
                best_ans = max(best_ans, height)
            else:
                # impossible for next
                break
    return best_ans


T = int(input())

for t in range(1, T+1):
    N = int(input())
    weights = list(map(int, input().split()))
    # test
    weights = list(range(1, N+1))
    # test
    weights.reverse()

    if N <= 100:
        ans = solve_easy(N, weights)
    else:
        ans = solve_easy(N, weights)
    print("Case #%d: %d" % (t, ans))

