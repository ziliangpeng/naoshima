
def handle_rev_str(conn):
    l = ord(recv_n(conn, 1))
    s = recv_n(conn, l)
    s = s[::-1]
    send_all(conn, s)
    conn.close()

def send_all(s, content):
    s.sendall(content)

def recv_n(s, n):
    ret = ''
    while len(ret) < n:
        ret += s.recv(n - len(ret))
    return ret
