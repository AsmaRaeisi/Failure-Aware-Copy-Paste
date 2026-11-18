from .lib import *
from .remove_redundant_points import remove_redundant_points
from.is_inclined_line import is_inclined_line
from.process_polygon import process_polygon


def update_existing_shapes_with_new_contour(existing_shapes, new_contour):

    # if we have one or 2 points in countour they won't form a polygon
    # Convert new_contour to a Polygon for spatial operations
    new_polygon = Polygon(new_contour)
    if not new_polygon.is_valid:
        new_polygon = new_polygon.buffer(0)  # Attempt to fix invalid geometry
    if not new_polygon.is_valid:
        print("Skipping new polygon due to invalid geometry.")
        return existing_shapes
    
    updated_shapes = []
    
    for shape in existing_shapes:
        existing_contour = np.array(shape['points'], dtype=np.float32)
        existing_polygon = Polygon(existing_contour)

        if not existing_polygon.is_valid:
            print(f"Invalid geometry detected for shape {shape['label']}. Attempting to fix...")
            existing_polygon = existing_polygon.buffer(0) #Create a zero-distance buffer around the geometry.

            if not existing_polygon.is_valid:
                print(f"Failed to fix geometry for shape {shape['label']}. Skipping.")
                continue
        
        #ensures that intersecting areas between two polygons are removed
        if existing_polygon.intersects(new_polygon):
            # Calculate the difference to remove intersecting areas from the existing shape
            difference_polygon = existing_polygon.difference(new_polygon)

            if not difference_polygon.is_empty:
                # Handle different geometry types appropriately
                if isinstance(difference_polygon, Polygon):
                    updated_shapes.extend(process_polygon(shape, difference_polygon))
                elif isinstance(difference_polygon, MultiPolygon):
                    # Handle MultiPolygon by adding each part separately
                    for poly in difference_polygon.geoms:
                        if poly.is_valid and not poly.is_empty:
                            updated_shapes.extend(process_polygon(shape, poly))
        else:
            # If no intersection, keep the original shape
            shape['points'] = np.round(np.array(shape['points'])).astype(float).tolist()
            # Ensure the contour is closed by adding the initial point to the end if necessary
            if shape['points'][0] != shape['points'][-1]:
                shape['points'].append(shape['points'][0])
            updated_shapes.append(shape)

    # Check for redundant points and remove them
    final_shapes = []
    for shape in updated_shapes:
        unique_points = remove_redundant_points(shape['points'])
        if len(unique_points) >= 3:
            # Ensure the contour is closed by adding the initial point to the end if necessary
            if unique_points[0] != unique_points[-1]:
                unique_points.append(unique_points[0])
            shape['points'] = unique_points

            # Check if all points have similar x or y values or form an inclined line
            x_values = [point[0] for point in shape['points']]
            y_values = [point[1] for point in shape['points']]
            if len(set(x_values)) > 1 and len(set(y_values)) > 1 and not is_inclined_line(shape['points']):
                final_shapes.append(shape)

    return final_shapes