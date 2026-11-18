from .lib import *
from .prepare_masks import prepare_masks

# ============================================================
# SCALE AUGMENTATION
# ============================================================

def apply_scale_augmentation(object_extracted, contour_points, kernel_size, scale_factor):


    object_resized = cv2.resize(object_extracted, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    contour_points = (contour_points) * scale_factor

    contour_points = contour_points.astype(np.int32)


    return prepare_masks(
        object_resized,
        contour_points,
        kernel_size
    )