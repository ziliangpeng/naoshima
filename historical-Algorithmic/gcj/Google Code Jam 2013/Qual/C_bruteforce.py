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

all_numbers = []

def ispalindrome(i):
    i = str(i)
    return i == ''.join(reversed(i))

def cal_all():
    for i in range(10000000):
        ii = i*i
        if ispalindrome(i) and ispalindrome(ii):
            all_numbers.append(ii)

class Solver(Work):
    def __init__(self, id, data_container):
        Work.__init__(self, id)
        self.data_container = data_container

    def _run(self):
        A, B = self.data_container.A, self.data_container.B
        a_index = bisect.bisect_left(all_numbers, A)
        b_index = bisect.bisect_right(all_numbers, B)
        return b_index - a_index


class DataContainer:
    def __init__(self, id):
        self.A, self.B = read_array(int)
        

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
    cal_all()
    # print all_numbers
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

