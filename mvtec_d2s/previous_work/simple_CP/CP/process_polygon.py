
from .lib import *


#ensure the contour is closed
def process_polygon(shape, polygon):
    # Round coordinates and convert to float
    rounded_coords = np.round(np.array(polygon.exterior.coords)).astype(float).tolist()
    # Ensure the contour is closed by adding the initial point to the end if necessary
    if rounded_coords[0] != rounded_coords[-1]:
        rounded_coords.append(rounded_coords[0])
    return [{
        'label': shape['label'],
        'points': rounded_coords,
        'group_id': shape['group_id'],
        'shape_type': 'polygon',
        'flags': shape['flags']
    }]
