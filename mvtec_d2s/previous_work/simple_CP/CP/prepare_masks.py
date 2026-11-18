from .lib import *

# ============================================================
# COMMON MASK PREPARATION
# ============================================================

def prepare_masks(object_image, contour_points, kernel_size):

    contour_points += 2 * kernel_size

    xn, yn, wn, hn = cv2.boundingRect(contour_points)

    mask_image_gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

    _, binary_mask_single_channel = cv2.threshold(mask_image_gray,1,255,cv2.THRESH_BINARY)

    padded_mask = np.pad(binary_mask_single_channel, pad_width=2 * kernel_size,mode='constant',constant_values=0)

    binary_mask_single_channel_crop = padded_mask[yn-kernel_size:yn-kernel_size+hn+(2*kernel_size), xn-kernel_size:xn-kernel_size+wn+(2*kernel_size)]

    mask_blur = cv2.GaussianBlur(padded_mask.astype(np.float32),(kernel_size, kernel_size),0)

    mask_blur = mask_blur[yn-kernel_size:yn-kernel_size+hn+(2*kernel_size),xn-kernel_size:xn-kernel_size+wn+(2*kernel_size)]

    blurred_edge_region = mask_blur.copy()

    blurred_edge_region[blurred_edge_region == 255] = 0.0

    blurred_edge_region[blurred_edge_region != 0.0] = 255

    outer_region = 255 - mask_blur.copy()

    outer_region[outer_region < 255] = 0.0

    inner_region = mask_blur.copy()

    inner_region[inner_region < 255] = 0.0

    object_padded = np.pad(
        object_image,
        pad_width=((2*kernel_size, 2*kernel_size), (2*kernel_size, 2*kernel_size), (0, 0)),
        mode='constant',
        constant_values=0
    )

    mask_image = object_padded[yn-kernel_size:yn-kernel_size+hn+(2*kernel_size), xn-kernel_size:xn-kernel_size+wn+(2*kernel_size)]

    x_min, y_min = np.min(contour_points, axis=0)

    contour_points[:, 0] -= x_min - kernel_size
    contour_points[:, 1] -= y_min - kernel_size

    return (
        contour_points,
        mask_image,
        binary_mask_single_channel_crop,
        blurred_edge_region,
        outer_region,
        inner_region
    )
