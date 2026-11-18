from .lib import *
from .find_edge_intersections import find_edge_intersections
from .insert_intersections_and_trim import insert_intersections_and_trim
#from .get_tote_bbx import get_tote_bbx
from .apply_gaussian_blending import apply_gaussian_blending


# ============================================================
# PLACEMENT FUNCTION
# ============================================================

def compute_overlap_ratio(
    x_offset,
    y_offset,
    binary_mask_single_channel_crop,
    background_json_path,
    background_width,
    background_height
):

    with open(background_json_path, "r") as f:
        data = json.load(f)

    existing_mask = np.zeros(
        (background_height, background_width),
        dtype=np.uint8
    )

    # Multiclass dataset: every valid polygon is an existing foreground instance.
    # Do not hard-code a class name such as "Object".
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 3:
            continue

        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(existing_mask, [pts], 255)

    new_mask = np.zeros(
        (background_height, background_width),
        dtype=np.uint8
    )

    h, w = binary_mask_single_channel_crop.shape

    # Clip the paste region so it stays within the background bounds.
    # The object can be larger than the background (or be offset past its
    # edges), which would otherwise raise a broadcast error.
    y0 = max(0, y_offset)
    x0 = max(0, x_offset)
    y1 = min(background_height, y_offset + h)
    x1 = min(background_width, x_offset + w)

    if y1 > y0 and x1 > x0:
        src_y0 = y0 - y_offset
        src_x0 = x0 - x_offset
        new_mask[y0:y1, x0:x1] = binary_mask_single_channel_crop[
            src_y0:src_y0 + (y1 - y0),
            src_x0:src_x0 + (x1 - x0)
        ]

    overlap_pixels = np.sum(
        (existing_mask > 0) &
        (new_mask > 0)
    )

    new_object_pixels = np.sum(new_mask > 0)

    if new_object_pixels == 0:
        return 1.0

    return overlap_pixels / new_object_pixels


def place_object(
        transform_type,
        contour_points,
        background_json_path,
        background_width,
        background_height,
        object_width,
        object_height,
        binary_mask_single_channel_crop,
        modified_background_image,
        output_image,
        blurred_edge_region,
        outer_region,
        inner_region,
        kernel_size, 
        x_offset=None,
        y_offset=None,
        rng=None,
        use_blending=True):


    if rng is None:
        rng = random

    if transform_type == 'Truncate':
        
        adjusted_contour = contour_points.copy()

        adjusted_contour[:, 0] += x_offset
        adjusted_contour[:, 1] += y_offset

        intersection_points = find_edge_intersections(adjusted_contour, background_width, background_height)

        adjusted_contour = insert_intersections_and_trim(adjusted_contour, intersection_points, background_width, background_height)


        if adjusted_contour.shape[0] > 0:  
            if not np.array_equal(adjusted_contour[0], adjusted_contour[-1]):
                adjusted_contour = np.vstack([adjusted_contour, adjusted_contour[0]])


        else:
            print("Warning: The contour is empty. Skipping this object.")
            return None, None, None, None

        adjusted_contour = adjusted_contour.astype(float)
    
    else:
        

        # Entire image is the placement region
        x_sub = max(0, background_width - object_width)
        y_sub = max(0, background_height - object_height)

        tote_x = 0
        tote_y = 0



        for attempt in range(50):

            x_offset = tote_x + int(rng.uniform(0, x_sub))
            y_offset = tote_y + int(rng.uniform(0, y_sub))

            overlap_ratio = compute_overlap_ratio(
                x_offset,
                y_offset,
                binary_mask_single_channel_crop,
                background_json_path,
                background_width,
                background_height
            )

            if overlap_ratio < 0.7:
                break


        # Safety clamp: if the object is larger than the tote along a dimension
        # (x_sub / y_sub == 0 and the tote origin pushes it past the image edge),
        # keep the paste region inside the image so the blend stays valid.
        x_offset = max(0, min(x_offset, background_width - object_width))
        y_offset = max(0, min(y_offset, background_height - object_height))



        adjusted_contour = contour_points.copy()
        adjusted_contour[:, 0] = (adjusted_contour[:, 0]) + x_offset
        adjusted_contour[:, 1] = (adjusted_contour[:, 1]) + y_offset


        adjusted_contour = adjusted_contour.astype(float)

    adjusted_contour = np.round(adjusted_contour).astype(float)
    


    T = modified_background_image.copy()

    if binary_mask_single_channel_crop.shape != T[y_offset:y_offset + object_height, x_offset:x_offset + object_width, :].shape[:2]:
        print(f"Skipping object due to shape mismatch: binary_mask_single_channel_crop: {binary_mask_single_channel_crop.shape}, target region: {T[y_offset:y_offset + object_height, x_offset:x_offset + object_width, :].shape}")
        return None, None, None, None 
    


    for c in range(0, 3):
        T[y_offset:y_offset + object_height, x_offset:x_offset + object_width, c] = \
            (1.0 -  binary_mask_single_channel_crop/255) * T[y_offset:y_offset + object_height, x_offset:x_offset + object_width, c] + \
            (( binary_mask_single_channel_crop/255) * output_image[:, :, c])




    if use_blending:
        # Current/default behavior: ring-based Gaussian blending.
        modified_background_image = apply_gaussian_blending(
            T,
            modified_background_image,
            blurred_edge_region,
            outer_region,
            inner_region,
            output_image,
            x_offset,
            y_offset,
            object_height,
            object_width,
            kernel_size
        )
    else:
        # Hard copy-paste: T already contains the object composited with the
        # original binary mask, with no Gaussian blur or ring blending.
        modified_background_image = T

    return (
        modified_background_image,
        adjusted_contour,
        x_offset,
        y_offset
    )