from Queue import Queue, Empty
from threading import Thread
import bisect

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

    def _run(self):
        A = self.data_container.A
        v = self.data_container.v
        v = sorted(v)

        lenv = len(v)
        ans = lenv
        add_cnt = 0
        i = 0

        if A == 1:
            return lenv

        while i < lenv:
            ans = min(add_cnt + lenv - i, ans)
            value = v[i]
            if A > value:
                A += value
                i += 1
            else:
                A = A * 2 - 1
                add_cnt += 1

        ans = min(ans, add_cnt)

        return ans
        pass # solve


class DataContainer:
    def __init__(self, id):
        self.A, self.N = read_array(int)
        self.v = read_array(int)
        pass # read data

#======================== EOF  =============================#

NUM_THREAD = 1
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

