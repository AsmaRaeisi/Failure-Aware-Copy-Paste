from .lib import cv2, np


def build_blending_regions(padded_mask, crop_slices, kernel_size, use_blending=True):
    """Build the binary crop and ring masks used by copy-paste blending.

    When ``use_blending`` is False, no Gaussian blur is computed. The returned
    auxiliary masks only preserve the expected shapes; the hard copy-paste path
    uses the binary mask directly.
    """
    binary_mask_crop = padded_mask[crop_slices]

    if not use_blending:
        inner_region = binary_mask_crop.astype(np.float32)
        outer_region = (255 - binary_mask_crop).astype(np.float32)
        blurred_edge_region = np.zeros_like(inner_region, dtype=np.float32)
        return (
            binary_mask_crop,
            blurred_edge_region,
            outer_region,
            inner_region,
        )

    mask_blur = cv2.GaussianBlur(
        padded_mask.astype(np.float32),
        (kernel_size, kernel_size),
        0,
    )[crop_slices]

    blurred_edge_region = mask_blur.copy()
    blurred_edge_region[blurred_edge_region == 255] = 0.0
    blurred_edge_region[blurred_edge_region != 0.0] = 255

    outer_region = 255 - mask_blur.copy()
    outer_region[outer_region < 255] = 0.0

    inner_region = mask_blur.copy()
    inner_region[inner_region < 255] = 0.0

    return (
        binary_mask_crop,
        blurred_edge_region,
        outer_region,
        inner_region,
    )
