from Queue import Queue, Empty
from threading import Thread
import bisect
import math


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

    def cal_area(self, n, r):
        return ((2 * r + 1) + (2 * r + 1 + 4 * n - 4)) * n / 2

    def _run(self):
        r, t = self.data_container.r, self.data_container.t

        left = 0
        right = 10 ** 19

        while left <= right:
            m = (left + right) / 2
            area = self.cal_area(m, r)
            if area > t:
                right = m - 1
            else:
                left = m + 1
        
        return right
        pass


class DataContainer:
    def __init__(self, id):
        self.r, self.t = read_array(int)
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

