from collections import defaultdict

# def solve(C, D, Xs, graphs, edges):
# nexts = []
# done = set()
# for d in graphs[0]:
#     nexts.append(d)
# while len(done) < C - 1:
#
# pass


def solve2(C, D, Xs, graphs, edges):
    time = {0: 0}
    edges_map = {}
    # Xs = [0] + Xs
    # time[0] = 0 # ordered by time, not id
    times = [0]
    # max_time = 0
    for i, x in enumerate(Xs):
        i = i + 1
        if x > 0:
            time[i] = x
            times.append(x)
    times.sort()

    Xs = [(-Xs[i], i + 1) for i in range(len(Xs)) if Xs[i] < 0]
    Xs.sort()
    for x, idx in Xs:
        # print 'xidx', x, idx
        prev_time = times[x - 1]
        now = prev_time + 1
        time[idx] = now
        # print 'now', id, now
        times.append(now)
        for d in graphs[idx]:
            if d in time and time[d] < now:
                edges_map[(d, idx)] = now - time[d]
                edges_map[(idx, d)] = now - time[d]
                break
        times.sort()

    lat = []
    for e in edges:
        if e in edges_map:
            lat.append(edges_map[e])
        else:
            u, v = e
            lat.append(max(1, abs(time[u] - time[v])))
    # print lat
    return " ".join(map(str, lat))


T = input()
for t in range(1, T + 1):
    C, D = map(int, raw_input().split())
    Xs = map(int, raw_input().split())
    # Xs = [x for x in Xs]
    graphs = defaultdict(list)
    edges = []
    for i in range(D):
        U, V = map(int, raw_input().split())
        U -= 1
        V -= 1
        graphs[U].append(V)
        graphs[V].append(U)
        if U > V:
            U, V = V, U
        edges.append((U, V))

    print("Case #%d: %s" % (t, solve2(C, D, Xs, graphs, edges)))
