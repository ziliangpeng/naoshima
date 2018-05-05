import math


def fuck_round(f):
    fr = math.floor(f)
    r = f - fr
    if r >= 0.5:
        return fr + 1
    else:
        return fr

# print(fuck_round(0.3))
# print(fuck_round(0.4))
# print(fuck_round(0.5))
# print(fuck_round(0.50000001))
# print(fuck_round(0.6))

T = int(input())
for t in range(1, 1 + T):
    N, L = map(int, input().split())
    C = list(map(int, input().split()))
    # print(N, C)
    ans = 0
    for c in C:
        ans += fuck_round(c*100/N)

    to_next_level = []
    for i, c in enumerate(C):
        rounded_perc = fuck_round(c*100/N)
        next_lvl = rounded_perc + 0.5
        need_for_next_lvl = math.ceil(N * next_lvl / 100)
        addition = fuck_round(need_for_next_lvl*100/N) - fuck_round(c*100/N)
        to_next_level.append((need_for_next_lvl - c, -addition, i, c))
    # print(to_next_level)

    to_next_level_from_0 = math.ceil(N * 0.005)
    to_next_level.sort()
    # print(to_next_level)
    tnl_i = 0

    remain = N - sum(C)
    # print('ans', ans)
    to_next_level_2 = []
    while remain > 0:
        if tnl_i < len(to_next_level):
            min_for_exist, addition, ci, c = to_next_level[tnl_i]
        else:
            min_for_exist = 10 ** 9
        min_for_new = to_next_level_from_0

        if min(min_for_exist, min_for_new) > remain:
            break

        if min_for_exist == min_for_new and \
                fuck_round(min_for_new*100/N) < fuck_round((C[ci]+min_for_exist)*100/N) - fuck_round(C[ci]*100/N):
            ans -= fuck_round(C[ci]*100/N)
            ans += fuck_round((C[ci] + min_for_exist)*100/N)
            remain -= min_for_exist
            tnl_i += 1

            c = C[ci] + min_for_exist
            rounded_perc = fuck_round(c*100/N)
            next_lvl = rounded_perc + 0.5
            need_for_next_lvl = math.ceil(N * next_lvl / 100)
            addition = fuck_round(need_for_next_lvl*100/N) - fuck_round(c*100/N)
            to_next_level_2.append((need_for_next_lvl - c, -addition, i, c))

            if tnl_i == len(to_next_level):
                to_next_level = to_next_level_2
                to_next_level.sort()
                tnl_i = 0
                to_next_level_2 = []
        elif min_for_exist < min_for_new:
            ans -= fuck_round(C[ci]*100/N)
            ans += fuck_round((C[ci] + min_for_exist)*100/N)
            remain -= min_for_exist
            tnl_i += 1

            c = C[ci] + min_for_exist
            rounded_perc = fuck_round(c*100/N)
            next_lvl = rounded_perc + 0.5
            need_for_next_lvl = math.ceil(N * next_lvl / 100)
            addition = fuck_round(need_for_next_lvl*100/N) - fuck_round(c*100/N)
            to_next_level_2.append((need_for_next_lvl - c, -addition, i, c))

            if tnl_i == len(to_next_level):
                to_next_level = to_next_level_2
                to_next_level.sort()
                tnl_i = 0
                to_next_level_2 = []
        else:
            ans += fuck_round(min_for_new*100/N)
            remain -= min_for_new
        # print('ans', ans)
        # need = math.ceil(N * 0.005)
        # if need > remain:
        #     need = remain
        # ans += fuck_round(need*100/N)
        # remain -= need

    print("Case #%d: %d" % (t, ans))