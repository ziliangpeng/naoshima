
class Point:
    def __init__(self, i, j):
        self.i = i
        self.j = j


class Segment:
    def __init__(self, start, end):
        self.start = min(start, end)
        self.end = max(start, end)


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = min(x1, x2)
        self.x2 = max(x1, x2)
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)


def is_segment_intersect(sa, sb):
    return max(sa.start, sb.start) < min(sa.end, sb.end)
    

def is_segment_inside(sa, sb):
    pass


def is_rect_intersect(ra, rb):
    pass


