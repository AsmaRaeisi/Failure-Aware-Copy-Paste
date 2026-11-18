
from .lib import *

# ============================================================
# DATA LOADER
# ============================================================

def load_background(background_image_path, background_json_path):

    initial_background_image = cv2.imread(background_image_path)

    with open(background_json_path, 'r') as background_json_file:
        initial_background_data = json.load(background_json_file)

    modified_background_image = initial_background_image.copy()

    modified_background_data = copy.deepcopy(initial_background_data)

    return (
        initial_background_image,
        modified_background_image,
        modified_background_data
    )
