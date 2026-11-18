from .lib import *

#find intersection points of contour segments with all edges and store them
from .get_line_intersection import get_line_intersection


def find_edge_intersections(contour, width, height):
    edges = [
        ((0, 0), (width, 0)),         # Top edge
        ((0, 0), (0, height)),        # Left edge
        ((width, 0), (width, height)), # Right edge
        ((0, height), (width, height)) # Bottom edge
    ]

    intersection_points = []
    for i in range(len(contour)):
        p1 = tuple(contour[i])
        p2 = tuple(contour[(i + 1) % len(contour)])

        for edge in edges:
            intersection = get_line_intersection(p1, p2, edge[0], edge[1])
            if intersection is not None:
                intersection_points.append((i, intersection))

    return intersection_points