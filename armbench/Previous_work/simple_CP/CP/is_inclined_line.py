

from .lib import *

#check whether points lie along a line or not
def is_inclined_line(points):
    # Check if all points lie along an inclined line (not purely horizontal or vertical)
    if len(points) > 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        for i in range(2, len(points)):
            x2, y2 = points[i]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                return False
        return True
    return False
