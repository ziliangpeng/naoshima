def solve(a):
    if a == 20:
        w, h = 6, 6
    elif a == 200:
        w, h = 15, 15

    m = [[0] * w for i in range(w)]
    # print 2 int
    for i in range(1, h, 3):
        for j in range(1, w, 3):
            while True:
                print(i+1, j+1) # 1-based
                readi, readj = list(map(int, input().split()))
                if readi == 0 and readj == 0:
                    # soved
                    return
                if readi == -1 and readj == -1:
                    # dead
                    return

                readi, readj = readi-1, readj-1 # convert to 0-based
                m[readi][readj] = 1
                if ok(m, i, j):
                    break

def ok(m, i, j):
    if not m[i-1][j-1]:
        return False
    if not m[i-1][j]:
        return False
    if not m[i-1][j+1]:
        return False
    if not m[i][j-1]:
        return False
    if not m[i][j]:
        return False
    if not m[i][j+1]:
        return False
    if not m[i+1][j-1]:
        return False
    if not m[i+1][j]:
        return False
    if not m[i+1][j+1]:
        return False
    return True


T = int(input())
for t in range(1, T+1):
    A = int(input())
    solve(A)


