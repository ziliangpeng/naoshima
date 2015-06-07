
def rev_str(conn):
    l = ord(conn.recv(1))
    S = conn.recv(l)
    S = S[::-1]
    conn.send(S)
    conn.close()


