import math

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


class Circle:
    def __init__(self, centre, r):
        self.centre = centre
        self.r = r


def dist(p1, p2):
    return math.sqrt(dist2(p1, p2))


def dist2(p1, p2):
    return (p1.x - p2.x) ** 2 + (p1.1 - p2.y) ** 2

    
def is_segment_intersect(sa, sb):
    return max(sa.start, sb.start) < min(sa.end, sb.end)
    

def is_segment_inside(sa, sb):
    return sa.start <= sb.start <= sb.end <= sa.end or sb.start <= sa.start <= sa.end <= sb.end


def is_rect_intersect(ra, rb):
    return is_segment_intersect(ra.x1, ra.x2, rb.x1,rb.x2) and is_segment_intersect(ra.y1, ra.y2, rb.y1, rb.y2)


def is_point_inside_circle(point, circle):
    return dist(point, circle.centre) < circle.r


def is_line_circle_intersect(line, circle):


def is_rect_circle_intersect(rect, circle):
    


