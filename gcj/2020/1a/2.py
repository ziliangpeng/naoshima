SZ = 50
mat = [[0 for i in range(50)] for j in range(50)]

for i in range(50):
    for j in range(i+1):
        if i == 0:
            mat[i][j] = 1
        elif j == 0:
            mat[i][j] = 1
        else:
            mat[i][j] = mat[i-1][j-1] + mat[i-1][j]

accu = [[0 for i in range(50)] for j in range(50)]
for i in range(50):
    for j in range(i+1):
        if i == 0:
            accu[i][j] = mat[i][j]
        elif j == 0:
            accu[i][j] = i + 1
        else:
            max_accu = max(accu[i][j-1], accu[i-1][j-1], accu[i-1][j])
            if max_accu < mat[i][j]:
                print('error ', i, j, max_accu, mat[i][j])
            accu[i][j] = max_accu + mat[i][j]
            # accu[i][j] =
mini = [[10000000000000 for i in range(50)] for j in range(50)]
for i in range(SZ):
    for j in range(i+1):
        if i == 0:
            mini[i][j] = mat[i][j]
        elif j == 0:
            mini[i][j] = mini[i-1][j] + mat[i][j]
        else:
            min_mini = min(mini[i][j-1], mini[i-1][j-1], mini[i-1][j])
            mini[i][j] = min_mini + mat[i][j]


for i in range(10):
    pass
    # print(mat[i])
    # print(accu[i][:10])
# print(mat[-1])

# def solve(N):
#     for i in range(SZ):
#         if max(accu[i]) > N:
#             start_i = i
#             break
#     for j in range(SZ):
#         if accu[i][j] > N:
#             start_j = j
#             break
#     i, j = start_i, start_j
#     N -= mat[i][j]
#     ans = [(i, j)]
#     while N > 0:
#         if i > 0 and accu[i-1][j] >= N:
#             i, j = i-1, j
#             ans.append((i, j))
#             N -= mat[i][j]
#         elif j > 0 and accu[i][j-1] >= N:
#             i, j = i, j-1
#             ans.append((i, j))
#             N -= mat[i][j]
#         elif i > 0 and j > 0 and accu[i-1][j-1] >= N:
#             i, j = i - 1, j - 1
#             ans.append((i, j))
#             N -= mat[i][j]
#         else:
#             print('error')
#
#     return ans

def solve2(N):
    start_i, start_j = -1, -1
    for i in range(SZ):
        for j in range(i+1):
            if accu[i][j] >= N and mini[i][j]<= N:
                start_i = i
                start_j = j
            if start_i != -1:
                break
        if start_i != -1:
            break

    i, j = start_i, start_j
    N -= mat[i][j]
    ans = [(i, j)]
    while N > 0:
        if i > 0 and accu[i-1][j] >= N and mini[i-1][j] <= N:
            i, j = i-1, j
            ans.append((i, j))
            N -= mat[i][j]
        elif j > 0 and accu[i][j-1] >= N and mini[i][j-1] <= N:
            i, j = i, j-1
            ans.append((i, j))
            N -= mat[i][j]
        elif i > 0 and j > 0 and accu[i-1][j-1] >= N and mini[i-1][j-1] <= N:
            i, j = i - 1, j - 1
            ans.append((i, j))
            N -= mat[i][j]
        else:
            print('error')

    return ans

def verify(mat, ans, N):
    if ans[0][0] != 0 or ans[0][1] != 0:
        print(ans[0], N, 'error3')
    v = 0
    for idx in ans:
        ii, jj = idx
        v += mat[ii][jj]
    if v != N:
        print('error2')

def test():
    for num in range(1, 1000001):
        ans = solve2(num)[::-1]
        verify(mat, ans, num)


def main():
    T = input()
    for t in range(1, T+1):
        N = input()
        ans = solve2(N)[::-1]
        verify(mat, ans, N)
        print("Case #%d:" % (t))
        for idx in ans:
            print("%d %d" % (idx[0] + 1, idx[1] + 1))

# for r in mini:
    # print r[:10]
# main()
test()
