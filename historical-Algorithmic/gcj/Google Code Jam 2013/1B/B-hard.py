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

    def get_heap(self, n, heap_cnt):
        i = 0
        while heap_cnt[i+1] <= n:
            i += 1
        return i, heap_cnt[i]


    def cal_which_heap(self, X, Y):
        X = abs(X)
        return (X+Y) / 2


    def C(self, n, k):
        ans = 1
        if k > n - k:
            k = n - k

        for i in range(k):
            ans *= n - i
            ans /= i + 1
        return ans


    def cal(self, n, k1, k2):
        ret = 0
        for k in range(k1, k2+1):
            if k > n:
                break
            if k < 0:
                continue
            ret += self.C(n, k)

        return ret


    def _run(self):
        n, X, Y = self.data_container.n, self.data_container.x, self.data_container.y
        if n == 0:
            return 0

        heap_cnt = [1]
        wrap = 2
        while heap_cnt[-1] <= 10 ** 7:
            heap_cnt.append(heap_cnt[-1] + wrap * 2 + 1)
            wrap += 2

        exist_heap, heap_sz = self.get_heap(n, heap_cnt)
        num_heap = self.cal_which_heap(X, Y)
        if num_heap > exist_heap + 1:
            return '0.000000'
        elif num_heap <= exist_heap:
            return '1.000000'
        else:
            max_height = exist_heap * 2 + 2
            wrap_length = heap_cnt[exist_heap+1] - heap_cnt[exist_heap]
            have = n - heap_sz
            if have - max_height >= Y + 1:
                return '1.000000'
            if have < Y+1:
                return '0.000000'

            #print 'have', have
            #print 'Y', Y
            #print 'max_height', max_height
            if Y + 1 > max_height:
                return '0.000000'
            return '%.6f' % (float(self.cal(have, Y+1, have)) / (1<<have))
        
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

