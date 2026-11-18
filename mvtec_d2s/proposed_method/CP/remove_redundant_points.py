
from .lib import *

#if we have more than 2 similar points in each polygan it will remove extra ones
def remove_redundant_points(points):
    # Find and keep only up to two pairs of similar points (for closing the contour)
    point_counts = {}
    for point in points:
        point_tuple = tuple(point)
        if point_tuple in point_counts:
            point_counts[point_tuple] += 1
        else:
            point_counts[point_tuple] = 1
    
    # Remove extra points, allowing only up to two occurrences of each point
    filtered_points = []
    for point in points:
        point_tuple = tuple(point)
        if point_counts[point_tuple] > 2:
            point_counts[point_tuple] -= 1
        else:
            filtered_points.append(point)

    distinct_points = []
    distinct_set = set()
    for point in filtered_points:
        point_tuple = tuple(point)
        if point_tuple not in distinct_set:
            distinct_points.append(point)
            distinct_set.add(point_tuple)
    
    return distinct_points