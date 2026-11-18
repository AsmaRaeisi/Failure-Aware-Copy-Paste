from .lib import *


# ============================================================
# OBJECT / HARD-EXAMPLE SELECTION
# ============================================================

# JSON-first cache, keyed by (annotation_folder, allowed_labels).
#
# This is required for the FN/FP folders produced by
# extract_hard_objects_train_set.ipynb:
#   images/                    = union of images containing FN and/or FP
#   json_fn_only/              = only images that contain an FN Object
#   json_fp_only_distractor/   = only images that contain an FP Distractor
#
# Therefore, sampling from images/ first is unsafe. We instead sample a valid
# JSON annotation first, confirm that it contains the requested label, and only
# then resolve and load its matching image from images/.
_JSON_CANDIDATE_CACHE = {}


# ------------------------------------------------------------
# QUALITY FILTER
# Used only when reject_tiny_objects=True. For hard FN objects and FP
# distractors, keep this False unless you intentionally want to remove tiny
# hard examples.
# ------------------------------------------------------------
MIN_OBJECT_AREA = 15000
MIN_OBJECT_SIDE = 40
MIN_THICKNESS = 40
MAX_ASPECT_RATIO = 4.0
MIN_SOLIDITY = 0.90


# ------------------------------------------------------------
# COMPLETENESS FILTER
# Used only when only_complete_objects=True.
# ------------------------------------------------------------
BORDER_MARGIN = 15
NEIGHBOR_MARGIN = 10


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _normalise_allowed_labels(allowed_labels):
    if isinstance(allowed_labels, str):
        allowed_labels = (allowed_labels,)

    if allowed_labels is None:
        return None

    return tuple(sorted(set(allowed_labels)))


def _valid_shapes_for_labels(object_data, allowed_labels):
    """Return polygon shapes that match the requested source labels."""
    allowed_set = None if allowed_labels is None else set(allowed_labels)

    valid_shapes = []
    for shape in object_data.get('shapes', []):
        if len(shape.get('points', [])) < 3:
            continue

        if allowed_set is not None and shape.get('label') not in allowed_set:
            continue

        valid_shapes.append(shape)

    return valid_shapes


def _get_json_candidates(object_annotations_folder, allowed_labels):
    """Index valid source JSONs before looking at the shared image folder.

    Each cached entry contains a JSON path and its parsed LabelMe data. A JSON
    is included only when it contains at least one polygon with an allowed
    label. Thus FN selection cannot accidentally choose an FP-only file, and
    FP selection cannot accidentally choose an FN-only file.
    """
    allowed_labels = _normalise_allowed_labels(allowed_labels)
    key = (os.path.abspath(object_annotations_folder), allowed_labels)

    cached = _JSON_CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached

    if not os.path.isdir(object_annotations_folder):
        raise FileNotFoundError(
            f"Annotation folder not found: {object_annotations_folder}"
        )

    candidates = []
    invalid_json_count = 0
    missing_label_count = 0

    for json_name in sorted(os.listdir(object_annotations_folder)):
        if not json_name.lower().endswith('.json'):
            continue

        json_path = os.path.join(object_annotations_folder, json_name)

        try:
            with open(json_path, 'r') as json_file:
                object_data = json.load(json_file)
        except Exception:
            invalid_json_count += 1
            continue

        if not _valid_shapes_for_labels(object_data, allowed_labels):
            missing_label_count += 1
            continue

        candidates.append((json_path, object_data))

    if not candidates:
        label_text = 'any polygon label' if allowed_labels is None else allowed_labels
        raise RuntimeError(
            "No valid source JSON files were found.\n"
            f"Annotation folder: {object_annotations_folder}\n"
            f"Required labels:   {label_text}\n"
            f"Invalid JSONs:     {invalid_json_count}\n"
            f"Without label:     {missing_label_count}"
        )

    _JSON_CANDIDATE_CACHE[key] = candidates
    return candidates


def _find_image_for_selected_json(object_images_folder, object_json_path, object_data):
    """Resolve the image only after a valid JSON has been selected.

    Prefer LabelMe's imagePath field because it records the actual image name.
    Fall back to the JSON stem for datasets where imagePath is absent or stale.
    """
    image_path_value = object_data.get('imagePath')

    if image_path_value:
        image_name = os.path.basename(str(image_path_value))
        direct_path = os.path.join(object_images_folder, image_name)
        if os.path.isfile(direct_path):
            return direct_path, image_name

        image_stem = os.path.splitext(image_name)[0]
        for ext in _IMAGE_EXTENSIONS:
            candidate = os.path.join(object_images_folder, image_stem + ext)
            if os.path.isfile(candidate):
                return candidate, os.path.basename(candidate)

    json_stem = os.path.splitext(os.path.basename(object_json_path))[0]
    for ext in _IMAGE_EXTENSIONS:
        candidate = os.path.join(object_images_folder, json_stem + ext)
        if os.path.isfile(candidate):
            return candidate, os.path.basename(candidate)

    return None, None


def _is_complete_object(pts, overlap_mask, tote_interior, img_shape):
    """Fast completeness test for one object, operating on its bbox ROI only."""
    h_img, w_img = img_shape[:2]
    x, y, w, h = cv2.boundingRect(pts)

    # Object touches image border.
    if x <= 0 or y <= 0 or (x + w) >= w_img or (y + h) >= h_img:
        return False

    local = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(local, [pts - [x, y]], -1, 255, thickness=cv2.FILLED)
    local_bool = local > 0

    # Object touches/crosses tote wall.
    if tote_interior is not None:
        tote_roi = tote_interior[y:y + h, x:x + w]
        if np.any(local_bool & (tote_roi == 0)):
            return False

    # Object is occluded/touching another object.
    ov_roi = overlap_mask[y:y + h, x:x + w]
    if np.any(local_bool & (ov_roi > 0)):
        return False

    return True


def _is_good_shape(shape):
    pts = np.array(shape['points'], dtype=np.int32)
    if len(pts) < 3:
        return False

    area = cv2.contourArea(pts)
    if area < MIN_OBJECT_AREA:
        return False

    x, y, w, h = cv2.boundingRect(pts)
    if w < MIN_OBJECT_SIDE or h < MIN_OBJECT_SIDE:
        return False

    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    thickness = min(rw, rh)
    length = max(rw, rh)

    if thickness < MIN_THICKNESS:
        return False
    if length / max(thickness, 1) > MAX_ASPECT_RATIO:
        return False

    hull_area = cv2.contourArea(cv2.convexHull(pts))
    if hull_area > 0 and (area / hull_area) < MIN_SOLIDITY:
        return False

    return True


def select_random_object(
    object_images_folder,
    object_annotations_folder,
    only_complete_objects=False,
    reject_tiny_objects=False,
    allowed_labels=None,
    rng=None,
):
    """Select one annotated source region using JSON-first sampling.

    Selection order
    ---------------
    1. Select a JSON from ``object_annotations_folder`` that contains the
       requested label.
    2. Resolve the corresponding image from ``object_images_folder``.
    3. Select one valid shape from that JSON.

    Parameters
    ----------
    allowed_labels:
        Which LabelMe labels can be selected. Use:
          ("Object",) for normal objects / FN hard positives.
          ("Distractor",) for FP distractors.
    """
    if rng is None:
        rng = random

    allowed_labels = _normalise_allowed_labels(allowed_labels)
    json_candidates = _get_json_candidates(
        object_annotations_folder,
        allowed_labels,
    )

    # A finite limit prevents an infinite loop if files are removed after the
    # cache is built or every candidate fails an optional quality filter.
    max_attempts = max(100, len(json_candidates) * 3)

    for _ in range(max_attempts):
        # IMPORTANT: choose the valid annotation first.
        object_json_path, object_data = rng.choice(json_candidates)

        # Only after selecting the JSON do we locate and load its image.
        object_image_path, object_image_name = _find_image_for_selected_json(
            object_images_folder,
            object_json_path,
            object_data,
        )

        if object_image_path is None:
            continue

        annotated_image = cv2.imread(object_image_path)
        if annotated_image is None:
            continue

        selectable_shapes = _valid_shapes_for_labels(object_data, allowed_labels)

        # Quality filtering is useful for complete normal objects, but should
        # normally be OFF for FN hard examples and FP distractors.
        if reject_tiny_objects:
            candidates = [s for s in selectable_shapes if _is_good_shape(s)]
        else:
            candidates = list(selectable_shapes)

        if not candidates:
            continue

        # Hard examples and distractors should usually not use completeness.
        if not only_complete_objects:
            return annotated_image, rng.choice(candidates), object_image_name

        img_h, img_w = annotated_image.shape[:2]

        # Completeness is computed using only the currently allowed labels.
        all_object_shapes = selectable_shapes

        count_map = np.zeros((img_h, img_w), dtype=np.uint8)
        for shape in all_object_shapes:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(
                mask,
                [np.array(shape['points'], dtype=np.int32)],
                1,
            )
            count_map += mask
        overlap_mask = (count_map >= 2).astype(np.uint8)

        tote_shape = next(
            (
                shape
                for shape in object_data.get('shapes', [])
                if shape.get('label') == 'Tote'
            ),
            None,
        )
        tote_interior = None
        if tote_shape is not None:
            tote_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(
                tote_mask,
                [np.array(tote_shape['points'], dtype=np.int32)],
                255,
            )
            tote_interior = cv2.erode(
                tote_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )

        rng.shuffle(candidates)
        for shape in candidates:
            pts = np.array(shape['points'], dtype=np.int32)
            if _is_complete_object(
                pts,
                overlap_mask,
                tote_interior,
                (img_h, img_w),
            ):
                return annotated_image, shape, object_image_name

    label_text = 'any polygon label' if allowed_labels is None else allowed_labels
    raise RuntimeError(
        "Could not select a usable source region after "
        f"{max_attempts} attempts.\n"
        f"Images:          {object_images_folder}\n"
        f"Annotations:     {object_annotations_folder}\n"
        f"Required labels: {label_text}"
    )
