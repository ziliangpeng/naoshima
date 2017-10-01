from Queue import Queue
from threading import Lock


class UniqueQueue(Queue):
    def __init__(self, n):
        self.q = Queue(n)
        self.s = set()
        self.put_lock = Lock()
        self.get_lock = Lock()

    def put(self, x):
        with self.put_lock:
            if x in self.s:
                print '%s already enqueued in UniqueQueue' % (str(x))
            else:
                self.s.add(x)
                self.q.put(x)

    def get(self):
        with self.get_lock:
            x = self.q.get()
            self.q.task_done()
            if x in self.s:
                self.s.remove(x)
            else:
                print 'Error: Queue key not in set'
            return x
