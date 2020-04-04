T = input()

for t in range(1, 1+T):
    s = raw_input()
    # print(s)
    ls = [(int(d), str(d)) for d in s]
    # print(ls)
    for num in range(9, 0, -1):
        new_ls = []
        index = 0
        while index < len(ls):
            if ls[index][0] == num:
                subs = ""
                while index < len(ls) and ls[index][0] == num:
                    subs += ls[index][1]
                    index += 1
                subs = '(' + subs + ')'
                new_ls.append((num-1, subs))
            else:
                new_ls.append(ls[index])
                index += 1
        ls = new_ls
        # print(ls)

    ans = "".join([s[1] for s in ls])
    print("Case #%d: %s" % (t, ans))
