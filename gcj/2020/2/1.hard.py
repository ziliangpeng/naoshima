T = input()


def asum(start, i, inc):
    end = start + (i - 1) * inc
    return (start + end) * i / 2


def max_serve(sz, start, inc=1):
    # number of customers served, pancackes used
    l, r = 0, sz
    while l <= r:
        m = (l + r) / 2
        end = start + (m - 1) * inc
        sum = (start + end) * m / 2
        if sum > sz:
            r = m - 1
        else:
            l = m + 1

    end = start + (r - 1) * inc
    return r, (start + end) * r / 2


# def solve(L, R):
#     if R > L:
#         l_s, l_u = max_serve(L, 1)
#         r_s, r_u = max_serve(R, 2)
#         return "N"
#     else:
#         l_s, l_u = max_serve(L, 1)
#         r_s, r_u = max_serve(R, 2)
#         if l_s <= r_s:
#             n_s = l_s * 2
#             return "%d %d %d" % (n_s, L - l_u, R - r_u)
#         elif l_s > r_s:
#             n_S = r_s * 2 + 1
#             n_s = l_s * 2
#             return "%d %d %d" % (n_s, L - l_u, R - r_u)

# def solve2(L, R):
#     i = 1
#     L = [L, 'l']
#     R = [R, 'r']
#     while L[0] >= i or R[0] >= i:
#         if L[0] < R[0]:
#             L, R = R, L
#         elif L[0] == R[0] and L[1] == 'r':
#             L, R = R, L
#         n, u = max_serve(L[0] - R[0], i)
#         L[0] -= u
#         i += n
#         assert L[0] >= R[0]
#         if L[0] >= i:
#             L[0] -= i
#             i += 1
#     if L[1] != 'l':
#         L, R = R, L
#     return "%d %d %d" % (i - 1, L[0], R[0])

# def solve3(L, R):
#     i = 1
#     while L >= i or R >= i:
#         if L < i:
#             R -= i
#         elif R < i:
#             L -= i
#         elif L >= R:
#             L -= i
#         else:
#             R -= i
#         i += 1
#     return "%d %d %d" % (i - 1, L, R)


def solve4(L, R):
    i = 1
    L = [L, 'l']
    R = [R, 'r']
    while L[0] >= i or R[0] >= i:
        # print L, R
        if L[0] < R[0]:
            L, R = R, L
        elif L[0] == R[0] and L[1] != 'l':
            L, R = R, L

        if L[0] == R[0]:
            L[0] -= i
            i += 1
            continue

        if L[0] - R[0] >= i:
            s, u = max_serve(L[0] - R[0], i)
            L[0] -= u
            i += s
            continue

        sl, ul = max_serve(L[0], i, 2)
        sr, ur = max_serve(R[0], i + 1, 2)
        # delta = L[0] - R[0]
        # sold = min(sl, sr, delta)
        sold = min(sl, sr)
        # def asum(start, i, inc):
        L[0] -= asum(i, sold, 2)
        R[0] -= asum(i + 1, sold, 2)
        i += sold * 2
        if L[0] >= i:
            L[0] -= i
            i += 1

    if L[1] != 'l':
        L, R = R, L
    return "%d %d %d" % (i - 1, L[0], R[0])

    # return str(L) + str(R)


for t in range(1, T + 1):
    L, R = map(int, raw_input().split())
    print("Case #%d: %s" % (t, solve4(L, R)))
