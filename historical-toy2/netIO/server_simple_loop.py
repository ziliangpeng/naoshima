import socket 
from net_common import handle_rev_str


HOST = ''
PORT = 5000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(1024)
print 'Server is up. Listening...'
while True:
    conn, addr = s.accept()
    print addr
    handle_rev_str(conn)
s.close()
