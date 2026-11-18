import json
import cv2
import numpy as np

json_file_path = "/media/asma/T7 Shield/cutpaste2026/exp17/anns/0F2HDP7QAS.json"
image_path = "/media/asma/T7 Shield/cutpaste2026/exp17/img_shadow/0F2HDP7QAS.jpg"

contour_output = "/home/asma/Desktop//contour-2.jpg"
diff_output = "/home/asma/Desktop//diff.jpg"


# Load annotation
with open(json_file_path, 'r') as json_file:
    annotation_data = json.load(json_file)


# Create empty image
image = np.zeros(
    (
        annotation_data["imageHeight"],
        annotation_data["imageWidth"],
        3
    ),
    dtype=np.uint8
)


# Colors
label_colors = {
    "Tote": (0, 0, 255),
    "Object": (0, 255, 0),
}


# Draw contours
for shape in annotation_data["shapes"]:

    label = shape["label"]

    points = np.array(shape["points"], dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    color = label_colors.get(label, (255, 255, 255))

    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=color,
        thickness=2
    )


# Save contour image
cv2.imwrite(contour_output, image)


# Create diff image
img = cv2.imread(image_path)
img2 = cv2.imread(contour_output)

result = cv2.subtract(img, img2)

cv2.imwrite(diff_output, result)

