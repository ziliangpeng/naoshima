from queue import Queue
from threading import Lock

from wrapt import synchronized


class UniqueQueue(Queue):
    def __init__(self, n):
        self.q = Queue(n)
        self.s = set()

    @synchronized(Lock())
    def put(self, x):
        if x in self.s:
            print('%s already enqueued in UniqueQueue' % (str(x)))
        else:
            self.s.add(x)
            self.q.put(x)

    @synchronized(Lock())
    def get(self):
        x = self.q.get()
        self.q.task_done()
        if x in self.s:
            self.s.remove(x)
        else:
            print('Error: Queue key not in set')
        return x
