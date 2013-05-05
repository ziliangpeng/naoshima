from Queue import Queue, Empty
from threading import Thread
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

class Solver(Work):
    def __init__(self, id, data_container):
        Work.__init__(self, id)
        self.data_container = data_container

    def _run(self):
        n, X, Y = self.data_container.n, self.data_container.x, self.data_container.y

        have = 0
        for bt in xrange(1<<n):
            highest = defaultdict(int)
            for i in range(n):
                x = 0
                height = highest[0]

                while True:
                    if height == 0:
                        break
                    elif highest[x-1] < highest[x] and highest[x+1] < highest[x]:
                        x_del = (1<<i) & bt and -1 or 1
                    elif highest[x-1] < highest[x]:
                        x_del = -1
                    elif highest[x+1] < highest[x]:
                        x_del = 1
                    else:
                        break

                    x += x_del
                    height -= 1

                highest[x] = height + 2
                highest[x-1] = height + 1
                highest[x+1] = height + 1

            if highest[X] > Y:
                have += 1

        return '%.6f' % (float(have) / (1<<n))
        pass # solve


class DataContainer:
    def __init__(self, id):
        self.n, self.x, self.y = read_array(int)
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

