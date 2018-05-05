from functools import reduce
from operator import mul
import random


def solve(N, L, words):
    poss = [ len(set([word[j] for word in words])) for j in range(L)]
    total_poss = reduce(mul, poss, 1)
    # print(words)
    # print(poss, total_poss)
    if total_poss == N:
        return '-'

    ans = [set() for _ in range(L)]
    for j in range(L):
        chosen = None
        col_j = [word[j] for word in words]
        for i in range(N):
            c = words[i][j]

            if col_j.count(c) < total_poss / poss[j]:
                # chosen = c
                ans[j].add(c)
        # if chosen is None:
        #     return '.'
        # else:
        #     ans += c
    # print('ans', ans)
    word_set = set(words)
    while True:
        select = ''.join([random.choice(list(ans[j])) for j in range(L)])
        if select not in word_set:
            return select


T = int(input())

for t in range(1, T+1):
    N, L = map(int, input().split())
    words = [input() for _ in range(N)]
    words = list(set(words))

    ans = solve(N, L, words)
    print("Case #%d: %s" % (t, ans))
