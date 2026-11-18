from .lib import *

def rotate_contour(contour, M): #applys the same rotation on contours but since the size of contour is smaller adds one for matrix multiplication
    contour_ones = np.hstack([contour, np.ones((contour.shape[0], 1), dtype=contour.dtype)])
    rotated_contour = M.dot(contour_ones.T).T
    return rotated_contour[:, :2]