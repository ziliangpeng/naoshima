
T = int(input())

debug = False

def cal(C, B):
    if B[0] == 0 or B[-1] == 0:
        return None
    state = [1] * C
    ans = []

    s = {
        0: '.',
        -1: '/',
        1: '\\'
    }
    step = 0
    while state != B:
        step += 1
        if debug:
            if step > 10:
                break
            print('state', state)
            print('B    ', B)

        diff = [B[i] - state[i] for i in range(C)]
        if debug:
            print('diff', diff)

        dir = [None] * C
        sum = 0
        for i in range(C):
            # if i == 0 or i == C - 1:
            #     dir[i] = 0

            # if state[i] == 0:
            #     dir[i] = 0
            # elif sum >= i + 1:
            if sum > 0:
                dir[i] = -1
                sum += diff[i]
            else:
                sum += diff[i]
                if sum >= 0:
                    dir[i] = 0
                else:
                    dir[i] = 1
                # if diff[i] == 0:
                #     dir[i] = 0
                # else:
                #     dir[i] = 1

        row = ''.join([s[d] for d in dir])
        if debug:
            print('dir', dir)
            print('row', row)

        for i in range(C):
            if state[i] > 0:
                if dir[i] == -1:
                    state[i-1] += 1
                    state[i] -= 1
                elif dir[i] == 1:
                    state[i+1] += 1
                    state[i] -= 1
        ans.append(row)

    ans.append('.' * C)
    return ans


for t in range(1, T+1):
    C = int(input())
    B = list(map(int, input().split()))

    ans = cal(C, B)
    if ans is None:
        print("Case #%d: IMPOSSIBLE" % (t))
    else:
        print("Case #%d: %d" % (t, len(ans)))
        print('\n'.join(ans))
        # print(ans)
