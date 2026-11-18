from .lib import *
from .rotate_image import rotate_image
from .rotate_contour import rotate_contour
from .prepare_masks import prepare_masks



# ============================================================
# ROTATION AUGMENTATION
# ============================================================

def apply_rotation_augmentation(object_extracted, contour_points, angle, kernel_size):

    object_extracted, M = rotate_image( object_extracted, angle)

    contour_points = rotate_contour(contour_points, M)

    contour_points = np.array(contour_points,dtype=np.int32)

    return prepare_masks(object_extracted,contour_points,kernel_size)




