from .lib import *

# ============================================================
# OBJECT EXTRACTION
# ============================================================

def extract_object(annotated_image, selected_shape):

    object_name = selected_shape['label']

    contour_points = np.array(selected_shape['points'], dtype=np.int32)

    mask = np.zeros_like(annotated_image)

    cv2.drawContours(mask, [contour_points], 0, (255, 255, 255), thickness=cv2.FILLED) 

    object_extracted = cv2.bitwise_and(annotated_image, mask)

    x, y, w, h = cv2.boundingRect(contour_points)

    return (
        object_name,
        contour_points,
        object_extracted,
        x, y, w, h
    )

