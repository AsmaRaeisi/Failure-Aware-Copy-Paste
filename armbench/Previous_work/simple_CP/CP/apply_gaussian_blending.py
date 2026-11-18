from .lib import *

# ============================================================
# BLENDING
# ============================================================

def apply_gaussian_blending(
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
        kernel_size):

    TB = cv2.GaussianBlur(T,(kernel_size, kernel_size),0)

    for c in range(0, 3):
        modified_background_image[y_offset:y_offset + object_height, x_offset:x_offset + object_width, c]  = \
            TB[y_offset:y_offset + object_height, x_offset:x_offset + object_width, c] * (blurred_edge_region/255) + \
            (outer_region/255) * modified_background_image[y_offset:y_offset + object_height, x_offset:x_offset + object_width, c] +\
            (inner_region/255) * output_image[:, :, c] 

    return modified_background_image
