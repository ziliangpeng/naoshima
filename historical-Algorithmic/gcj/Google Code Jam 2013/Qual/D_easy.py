from Queue import Queue, Empty
from threading import Thread
import bisect
from collections import defaultdict

def read_array(convertor=None):
    ret = raw_input().split()
    if convertor: ret = map(convertor, ret)
    return ret

work_queue = Queue()
result_queue = []

class Work:
    def __init__(self, id):
        self.id = id

    def solve(self):
        result = self._run()
        result_queue.append((self.id, result))

#======================== SOF  =============================#

def name(i):
    return '%02d' % (i)


class Solver(Work):
    def __init__(self, id, data_container):
        Work.__init__(self, id)
        self.data_container = data_container

    def _run(self):
        data_container = self.data_container
        K, N = data_container.K, data_container.N
        keys = data_container.keys
        needs = data_container.needs
        gets = data_container.gets

        #init
        startswith = defaultdict(int)
        for key in keys:
            startswith[key] += 1
        dp = [None] * (1<<N)
        have = [None] * (1<<N)
        dp[0] = ''
        have[0] = startswith

        #dp
        for bt in xrange(1<<N):
            for i in range(N):
                if bt & (1<<i):
                    prev = bt^(1<<i)
                    if dp[prev] != None:
                        if dp[bt] == None or dp[prev] + name(i) < dp[bt]:
                            if  have[prev][needs[i]] > 0:
                                dp[bt] = dp[prev] + name(i)
                                if have[bt] == None:
                                    have[bt] = defaultdict(int)
                                    have[bt].update(have[prev])
                                    have[bt][needs[i]] -= 1
                                    for get in gets[i]:
                                        have[bt][get] += 1

        ans = []
        ans_str = dp[(1<<N)-1]
        if ans_str == None:
            return 'IMPOSSIBLE'
        else:
            for i in range(0, len(ans_str), 2):
                s_val = ans_str[i:i+2]
                i_val = int(s_val)
                ans.append(str(i_val+1))
            return ' '.join(ans)


class DataContainer:
    def __init__(self, id):
        self.K, self.N = read_array(int)
        self.keys = read_array(int)

        self.needs = []
        self.gets = []
        for i in range(self.N):
            v = read_array(int)
            self.needs.append(v[0])
            self.gets.append(v[2:])

#======================== EOF  =============================#

NUM_THREAD = 3
class Executor(Thread):
    def run(self):
        while True:
            try:
                work = work_queue.get_nowait()
                work.solve()
                work_queue.task_done()
            except Empty:
                return


def main():
    T = input()
    executors = [Executor() for i in range(NUM_THREAD)]
    for t in range(T):
        data_container = DataContainer(t)
        work_queue.put(Solver(t, data_container))
    for executor in executors:
        executor.start()

    for executor in executors:
        executor.join()
    global result_queue
    result_queue = sorted(result_queue)
    for id, result in result_queue:
        print 'Case #%d:' % (id+1),
        print str(result)


if __name__ == '__main__':
    main()

