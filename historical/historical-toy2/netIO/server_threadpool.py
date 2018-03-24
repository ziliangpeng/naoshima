import socket 
from threading import Thread
from time import sleep
from net_common import handle_rev_str
from Queue import Queue, Empty
import signal
import sys

HOST = 'localhost'
PORT = 5000
THREAD_POOL_SIZE = 12

def sigterm_handler(_signo, _stack_frame):
    global end
    end = True
    global s
    s.close()
    sys.exit(0)


signal.signal(signal.SIGINT, sigterm_handler)


q = Queue()
end = False

class Worker(Thread):
    def run(self):
        while not end:
            try:
                conn = q.get(timeout=0.3)
                handle_rev_str(conn)
            except Empty:
                pass


def accept_thread(s):
    s.settimeout(0.3)
    while not end:
        try:
            conn, addr = s.accept()
        except socket.timeout:
            continue
        print addr
        conn.setblocking(True)
        q.put(conn)

workers = []
for _ in range(THREAD_POOL_SIZE):
    workers.append(Worker())
    workers[-1].start()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(1024)
Thread(target=accept_thread, args=(s,)).start()

while not end:
    sleep(60)
s.close()
