
from .lib import *


## truncation

#calculate intersection points of two line segments
# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

def get_line_intersection(p1, p2, q1, q2): 
    s1_x, s1_y = p2[0] - p1[0], p2[1] - p1[1]
    s2_x, s2_y = q2[0] - q1[0], q2[1] - q1[1]

    denom = (-s2_x * s1_y + s1_x * s2_y)
    if denom == 0:
        return None

    s = (-s1_y * (p1[0] - q1[0]) + s1_x * (p1[1] - q1[1])) / denom
    t = (s2_x * (p1[1] - q1[1]) - s2_y * (p1[0] - q1[0])) / denom

    if 0 <= s <= 1 and 0 <= t <= 1:
        intersection_x = p1[0] + (t * s1_x)
        intersection_y = p1[1] + (t * s1_y)
        return [int(intersection_x), int(intersection_y)]

    return None