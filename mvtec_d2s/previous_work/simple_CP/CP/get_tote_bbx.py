from .lib import *

def get_tote_bbx(json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    shapes = data.get('shapes', [])

    for shape in shapes:
        if shape.get('label', '') == 'Tote':
            points = shape.get('points', [])
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]

            return {
                'x': min(x_coords),
                'y': min(y_coords),
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords),
            }

    # If "Tote" label is not found, return a default bounding box covering the entire image
    return {
        'x': 0,
        'y': 0,
        'width': float('1000'),  # Set to a large value to cover the entire width
        'height': float('1000'),  # Set to a large value to cover the entire height
    }