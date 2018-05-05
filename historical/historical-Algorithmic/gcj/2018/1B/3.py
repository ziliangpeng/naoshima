import math

visited = {}

def recur(R1, R2, G, depth, M, tried):
    if depth > 8:
        return 0
    # tg = '.'.join(map(str, G))
    # if tg in visited:
    #     return visited[tg]

    best = G[0]
    for i in range(M):
        # if tg in visited:
        #     return visited[tg]
        r1 = R1[i]
        r2 = R2[i]
        if (r1, r2) in tried:
            continue
        if r1 == 0 or r2 == 0:
            continue
        if r1 == i or r2 == i:
            continue
        if G[r1] > 0 and G[r2] > 0:
            convert = min(G[r1], G[r2])

            # gg = G[:]
            tried.add((r1, r2))
            for use in range(1, convert + 1):
                # gg = G[:]
                # gg[i] += use
                # gg[r1] -= use
                # gg[r2] -= use
                G[i] += 1
                G[r1] -= 1
                G[r2] -= 1
                ans = recur(R1, R2, G, depth + 1, M, tried)
                if ans > best:
                    best = ans
            tried.remove((r1, r2))
            G[i] -= convert
            G[r1] += convert
            G[r2] += convert
    # visited[tg] = best
    return best


T = int(input())
for t in range(1, 1 + T):
    M = int(input())
    R1 = []
    R2 = []
    for m in range(M):
        ri1, ri2 = list(map(int, input().split()))
        R1.append(ri1-1)
        R2.append(ri2-1)
    G = list(map(int, input().split()))

    best = recur(R1, R2, G, 0, M, set())
    print("Case #%d: %d" % (t, best))

