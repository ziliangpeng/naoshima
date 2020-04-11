def input(t=int):
    return map(t, raw_input().split())

def get_data():
    pass

def solve():
    pass

def main():
    T = input()

    for t in range(1, T+1):
        x = get_data()

        ans = solve(x)

        print("Case #%d: %s" % (t, ans))

main()
