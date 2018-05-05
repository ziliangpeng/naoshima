T = int(input())

for t in range(1, T+1):
    N = int(input())
    sold = set()
    demand = [0 for _ in range(N)]
    for n in range(N):
        pref = list(map(int, input().split()))[1:]
        select, least_demand = -1, N + 10
        for p in pref:
            demand[p] += 1
        for p in pref:
            if p not in sold and demand[p] < least_demand:
                select = p
                least_demand = demand[p]
                # break
        print(select)
        sold.add(select)


