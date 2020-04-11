def solve(mat):
    score = 0
    while True:
        # print(mat)
        interest = 0
        for c in mat:
            for lvl in c:
                interest += lvl
        score += interest

        # eleminate
        I = len(mat)
        J = len(mat[0])
        changed = False
        mat2 = [[0 for j in range(J)] for i in range(I)]
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == 0:
                    continue
                comp_sum = 0
                comp_cnt = 0
                # up
                ii, jj = i-1, j
                while ii >= 0:
                    if mat[ii][jj] != 0:
                        comp_cnt += 1
                        comp_sum += mat[ii][jj]
                        break
                    ii -= 1

                # left
                ii, jj = i, j-1
                while jj >= 0:
                    if mat[ii][jj] != 0:
                        comp_cnt += 1
                        comp_sum += mat[ii][jj]
                        break
                    jj -= 1

                # right
                ii, jj = i, j+1
                while jj < J:
                    if mat[ii][jj] != 0:
                        comp_cnt += 1
                        comp_sum += mat[ii][jj]
                        break
                    jj += 1

                # down
                ii, jj = i+1, j
                while ii < I:
                    if mat[ii][jj] != 0:
                        comp_cnt += 1
                        comp_sum += mat[ii][jj]
                        break
                    ii += 1

                if comp_cnt != 0 and mat[i][j] < comp_sum * 1.0 / comp_cnt:
                    # print(i, j, comp_cnt, comp_sum)
                    changed = True
                    mat2[i][j] = 0
                else:
                    mat2[i][j] = mat[i][j]
        mat = mat2
        if not changed:
            break

    return score



    pass


T = input()
for t in range(1, T+1):
    R, C = map(int, raw_input().split())
    mat = []
    for r in range(R):
        c = map(int, raw_input().split())
        mat.append(c)
    print("Case #%d: %d" % (t, solve(mat)))
    # print("")
