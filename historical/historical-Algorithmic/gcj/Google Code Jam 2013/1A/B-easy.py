from Queue import Queue, Empty
from threading import Thread
import bisect
from datetime import datetime

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

class Solver(Work):
    def __init__(self, id, data_container):
        Work.__init__(self, id)
        self.data_container = data_container

    def dfs(self, i, e, r, max_e, v):
        if i == len(v):
            return 0

        ans = 0
        for use in range(e+1):
            gain = use * v[i]
            remain_e = min(max_e, e - use + r)
            gain += self.dfs(i+1, remain_e, r, max_e, v)
            ans = max(ans, gain)

        return ans

    def _run(self):
        e, r, n = self.data_container.e, self.data_container.r, self.data_container.n
        v = self.data_container.v
        if r > e:
            r = e

        return self.dfs(0, e, r, e, v)
        pass


class DataContainer:
    def __init__(self, id):
        self.e, self.r, self.n = read_array(int)
        self.v = read_array(int)
        pass # read data

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

