import math

T = int(input())

for t in range(1, T + 1):
    N, P = list(map(int, input().split()))
    cookies = []
    ws = []
    hs = []
    fix_size = 0.0
    for n in range(N):
        w, h = list(map(int, input().split()))
        # make w smaller
        if w > h:
            w, h = h, w
        cookies.append((w, h))
        ws.append(w)
        hs.append(h)
        fix_size += w * 2 + h * 2

    if sum(ws) * 2 + fix_size <= P:
        # simple solution
        largest = 0.0
        for i in range(N):
            w, h = ws[i], hs[i]
            largest += math.sqrt(w * w + h * h) * 2

        if fix_size + largest > P:
            ans = P
        else:
            ans = fix_size + largest
    else:
        # harder
        if len(set(ws)) == 1:
            # easy data
            edge_size = ws[0]
            cut_count = ((P - fix_size) / 2) // edge_size
            w = ws[0]
            h = hs[0]

            largest = math.sqrt(w*w + h*h) * 2 * cut_count
            if fix_size + largest > P:
                ans = P
            else:
                ans = fix_size + largest
        else:
            cookies.sort(reverse=True)
            cut_count = 0
            extra = 0
            for i in range(N):
                added = cookies[i][0] * 2
                if fix_size + extra + added <= P:
                    cut_count += 1
                    extra += added
                else:
                    break

            largest = 0
            for i in range(cut_count):
                w = cookies[i][0]
                h = cookies[i][1]
                largest += math.sqrt(w*w+h*h) * 2
            if fix_size + largest > P:
                ans = P
            else:
                ans = fix_size + largest

    print("Case #%d: %.8f" % (t, ans))