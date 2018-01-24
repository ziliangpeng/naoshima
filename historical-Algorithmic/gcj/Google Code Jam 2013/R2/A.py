from collections import defaultdict

def read_array(convertor=None):
    ret = raw_input().split()
    if convertor: ret = map(convertor, ret)
    return ret

MOD = 1000002013

def main():
    T = input()
    
    for _ in range(T):
        do = defaultdict(int)
        de = defaultdict(int)
        N, M = read_array(int)
        gain1 = 0
        for m in range(M):
            o, e, p = read_array(int)
            do[o] += p
            de[e] += p
            gain1 += (N + (N + 1 - (e - o))) * (e - o) / 2 * p

        lo = [[k, v] for k, v in do.items()]
        le = [[k, v] for k, v in de.items()]
        lo.sort()
        le.sort()

        gain2 = 0
        for e_pair in le:
            end, count = e_pair
            while count:
                select_pair = None
                for oi in range(len(lo)):
                    o_pair = lo[oi]
                    if o_pair[0] <= end and o_pair[1] > 0:
                        select_pair = o_pair

                start = select_pair[0]
                start_c = select_pair[1]
                min_count = min(count, start_c)

                gain2 += (N + (N + 1 - (end - start))) * (end - start) / 2 * min_count

                count -= min_count
                select_pair[1] -= min_count

        print 'Case #%d: %d' % (_+1, (gain1 - gain2) % MOD)




        
    pass



main()
