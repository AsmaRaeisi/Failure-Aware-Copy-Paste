
from .lib import *

# keeps all the contour points within image range and also insert intersection points, so we will have a closed loop
def insert_intersections_and_trim(contour, intersections, width, height):
    new_contour = []
    insertions = {pos: [] for pos, _ in intersections}

    for pos, inter in intersections:
        insertions[pos].append(inter)

    for i in range(len(contour)):
        point = contour[i]
        if 0 <= point[0] <= width and 0 <= point[1] <= height:
            new_contour.append(point)
        if i in insertions:
            new_contour.extend(insertions[i])

    return np.array(new_contour)