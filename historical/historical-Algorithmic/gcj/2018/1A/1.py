T = int(input())

CHOP = '@'

def count_chop_col(cake, c):
    return list(map(lambda r:r[c], cake)).count(CHOP)

def count_chop_row(cake, r):
    return cake[r].count(CHOP)


def solve(R, C, H, V, cake):
    total_chop = sum([count_chop_row(cake, i) for i in range(R)])
    if total_chop == 0:
        return True

    if total_chop % (H + 1) != 0:
        return False
    if total_chop % (V + 1) != 0:
        return False
    if total_chop % ((V+1) * (H+1)) != 0:
        return False

    per_row_chop = total_chop / (H + 1)
    per_col_chop = total_chop / (V + 1)
    per_cell_chop = total_chop / ((H+1) * (V+1))

    h_cuts = [0]
    v_cuts = [0]

    for c in range(C):
        prev_c = v_cuts[-1]
        col_chop = 0
        for cc in range(prev_c, c+1):
            col_chop += count_chop_col(cake, cc)
        # print('col chop, c', col_chop, c)
        if col_chop == per_col_chop:
            v_cuts.append(c+1)
        elif col_chop > per_col_chop:
            return False
    if len(v_cuts) != V + 2:
        return False

    for r in range(R):
        prev_r = h_cuts[-1]
        row_chop = 0
        for rr in range(prev_r, r+1):
            row_chop += count_chop_row(cake, rr)
        if row_chop == per_row_chop:
            h_cuts.append(r+1)
        elif row_chop > per_row_chop:
            return False
    if len(h_cuts) != H + 2:
        return False

    for vci in range(1, len(v_cuts)):
        for hci in range(1, len(h_cuts)):
            start_col = v_cuts[vci-1]
            end_col = v_cuts[vci]
            start_row = h_cuts[hci-1]
            end_row = h_cuts[hci]

            this_cell_chop = 0
            for c in range(start_col, end_col):
                for r in range(start_row, end_row):
                    if cake[r][c] == CHOP:
                        this_cell_chop += 1
            # print("per cel chop", per_cell_chop)
            # print('sc, ec, sr, er', start_col, end_col, start_row, end_row)
            # print('vci, hci, this chop', vci, hci, this_cell_chop)
            if this_cell_chop != per_cell_chop:
                return False

    return True



for t in range(1, T+1):
    R, C, H, V = list(map(int, input().split()))
    cake = [input() for _ in range(R)]
    ans = solve(R, C, H, V, cake)
    print("Case #%d: " % (t), ans and "POSSIBLE" or "IMPOSSIBLE")
