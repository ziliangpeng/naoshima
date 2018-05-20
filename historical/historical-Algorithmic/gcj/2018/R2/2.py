T = int(input())

debug = False

# def cal(R, B):
#     pass
#     dp = [[[0 for b in range(B+1)] for r1 in range(R+1)] for r2 in range(R+1)]
#
#     # dp[i][j][k] = max b per person is i,
#     dp[0][0] = 0
#     for r in range(R):
#         for b in range(B):

def s(r):
    return int((0 + r) * (r + 1) / 2)

def dfs(r, b, used_r, used_b, R, B, juggler):
    need_more_r = s(r)
    need_more_b = b * (r + 1)
    if need_more_b + used_b > B or need_more_r + used_r > R:
        return -1

    best = juggler + r + 1
    if debug:
        print(r, b, used_r, used_b, juggler, 'best1', best)
    for r2 in range(r+1):
        ans = dfs(r2, b+1, used_r + need_more_r, used_b + need_more_b, R, B, juggler + r + 1)
        if ans > best:
            best = ans
    if debug:
        print(r, b, used_r, used_b, juggler, 'best2', best)
    return best


def cal(R, B):
    pass
    best = 0
    for r in range(0, 100):
        # person having 0 blue have max of r red
        ans = dfs(r, 0, 0, 0, R, B, 0)
        if ans > best:
            best = ans
    return best


for t in range(1, T+1):
    R, B = list(map(int, input().split()))

    ans = cal(R, B) - 1
    print("Case #%d: %d" % (t, ans))
