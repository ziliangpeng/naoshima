import socket
import random

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
HOST = 'localhost'
PORT = 5000
conn = socket.create_connection((HOST, PORT))
l = random.randrange(3, 16)
s = ''.join([random.choice(ALPHA) for _ in range(l)])
conn.sendall(chr(l))
conn.sendall(s)
rev = conn.recv(l)
if s != rev[::-1]:
    print 'wrong'
