import socket
import random
from net_common import send_all, recv_n

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
HOST = 'server'
PORT = 5000
conn = socket.create_connection((HOST, PORT))
l = random.randrange(3, 128) # must fit in single byte
s = ''.join([random.choice(ALPHA) for _ in range(l)])
send_all(conn, chr(l))
send_all(conn, s)
rev = recv_n(conn, l)
if s != rev[::-1]:
    print 'wrong'
