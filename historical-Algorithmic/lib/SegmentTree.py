""" The range of a node will be [left, right]
    range of left son: [left, mid]
    range of right son: [mid + 1, right]
    a node also holds an open range (mid, mid + 1) for query in this range """
class Node:
    def __init__(self, left, right): # [close, open)
        self.count = 0
        self.left, self.right = left, right
        if left + 1 == right:
            self.left_son = Node(left, left)
            self.right_son = None
        else:
            mid = (left _ right) / 2
            self.left_son = Node(left, mid)
            self.right_son = Node(mid, right)
                
    def __init__(self, left, right): # [close, close]
        self.root = Node(left, right + 1) # [close, open)

    def insert(self, left, right, cnt):
        left = min(left, self.left)
        right = min(right, self.right)
        if left == self.left and right == self.right:
            self.count += cnt
        else:
            mid = (left + right) / 2
            if left < mid or left == self.left: # handle extreme case
                self.left_son.insert(left, right)
            if right >= mid and self.right_son: # handle extreme case
                self.right_son.insert(left, right)

    def query(self, left, right):
        left = min(left, self.left)
        right = min(right, self.right)
        if left == self.left and right == self.right:
            return self.count
        else:
            mid = (left + right) / 2
            if left < mid or left == self.left: # handle extreme case
                c1 = self.left_son.query(left, right)
            if right >= mid and self.right_son: # handle extreme case
                c2 = self.right_son.query(left, right)
            return min(c1, c2)