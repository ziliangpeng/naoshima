
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

    def insert(self, val):
        n = Node(val)
        n.next = self.next
        self.next = n

class LinkedList:
    def __init__(self):
        self.head = Node(None)
        self.tail = Node(None)
        self.head.next = self.tail

    def first(self):
        if self.head.next == self.tail:
            return None
        else:
            return self.head.next

    def insert_head(self, val):
        n = Node(val)
        n.next = self.head.next
        self.head.next = n

    def insert_at_i(self, val, i):
        ptr = self.head
        for i in range(i):
            ptr = ptr.next
            if ptr == self.tail:
                raise IndexError()
        n = Node(val)
        n.next = ptr.next
        ptr.next = n

    def index(self, i):
        ptr = self.head
        for _ in range(i + 1):
            ptr = ptr.next
            if ptr == self.tail:
                raise IndexError("out of bound")
        return ptr


    def print_all(self):
        elem = []
        ptr = self.head.next
        while ptr != self.tail:
            elem.append(ptr.val)
            ptr = ptr.next
        print(' '.join(map(str, elem)))



ll = LinkedList()
ll.insert_head(5)
ll.print_all() # 5
ll.insert_head(7)
ll.print_all() # 7 5
ll.first().insert(8)
ll.print_all() # 7 8 5
ll.index(2).insert(3)
ll.print_all() # 7 8 5 3
try:
    ll.index(10)
    raise BaseException("bug")
except IndexError:
    pass

ll.insert_at_i(10, 0)
ll.print_all() # 10 7 8 5 3
ll.insert_at_i(11, 2)
ll.print_all() # 10 7 11 8 5 3
