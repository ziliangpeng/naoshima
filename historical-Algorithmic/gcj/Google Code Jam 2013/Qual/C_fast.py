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

all_numbers = set()

def ispalindrome(i):
    i = str(i)
    return i == ''.join(reversed(i))

def make_palindrome(s):
    return ''.join(reversed(s)) + s, ''.join(reversed(s[1:])) + s 

def check(p):
    p = int(p)
    pp = p * p
    if ispalindrome(pp):
        all_numbers.add(pp)

def dfs(sb, count1, l):
    if l > 25:
        return
    if count1 > 6:
        return

    if sb:
        p1, p2 = make_palindrome(sb)
        check(p1)
        check(p2)
    if count1 <= 2:
        for i in range(len(sb)+1):
            sb2 = sb[:i] + '2' + sb[i:]
            p1, p2 = make_palindrome(sb2)
            check(p1)
            check(p2)

    dfs(sb + '0', count1, l + 1)
    dfs(sb + '1', count1 + 1, l + 1)

def cal_all():
    global all_numbers
    all_numbers.add(0)
    all_numbers.add(1)
    all_numbers.add(4)
    all_numbers.add(9)

    dfs('', 0, 0)

    all_numbers = list(sorted(all_numbers))

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
    # print datetime.now()
    cal_all()
    # print datetime.now()
    # print all_numbers
    # print len(all_numbers)
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

