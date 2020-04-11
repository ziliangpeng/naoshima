def common_prefix(ps):
    prefixes = []
    for p in ps:
        if p[0] != '*':
            prefixes.append(p[:p.find('*')])
    if len(prefixes) == 0:
        return ""

    longest = prefixes[0]
    for prefix in prefixes:
        if len(prefix) > len(longest):
            longest = prefix

    for prefix in prefixes:
        if not longest.startswith(prefix):
            return None

    return longest


def trim(p):
    while p[0] != '*':
        p = p[1:]
    while p[-1] != '*':
        p = p[:-1]
    return p


def solve(patterns):
    com_pre = common_prefix(patterns)
    com_suf = common_prefix([p[::-1] for p in patterns])
    if com_pre == None or com_suf == None:
        return '*'
    com_suf = com_suf[::-1]

    trimmed_ps = map(trim, patterns)
    trimmed_ps = map(lambda p: p.replace('*', ''), trimmed_ps)
    concat = ''.join(trimmed_ps)

    return com_pre + concat + com_suf

def main():
    T = input()

    for t in range(1, T+1):
        N = input()
        patterns = []
        for n in range(N):
            p = raw_input()
            patterns.append(p)

        print("Case #%d: %s" % (t, solve(patterns)))

main()
