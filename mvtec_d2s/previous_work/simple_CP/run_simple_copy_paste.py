
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from CP.random_utils import stable_seed


# ============================================================
# CONFIGURATION: EDIT ONLY THESE PATHS / SETTINGS IF NEEDED
# ============================================================

# The same dataset is used as both the donor-object set and background set.
OBJECT_IMAGES_FOLDER = Path(
    "**/train/images"
)
OBJECT_ANNOTATIONS_FOLDER = Path(
    "**/train/jsons"
)

BACKGROUND_IMAGES_FOLDER = Path(
    "**/train/images"
)
BACKGROUND_ANNOTATIONS_FOLDER = Path(
    "**/train/jsons"
)

OUTPUT_ROOT = Path(
    "**/simpleCP/"
)
OUTPUT_IMAGES_FOLDER = OUTPUT_ROOT / "img"
OUTPUT_ANNOTATIONS_FOLDER = OUTPUT_ROOT / "json"
OUTPUT_METADATA_FOLDER = OUTPUT_ROOT / "metadata"

MASTER_SEED = 42

# One shared class map is the authoritative list of valid dataset classes.
# Keep the same classes.json for real train/val/test, Simple Copy-Paste, and
# later YOLO conversion. The list order defines the fixed YOLO class IDs.
CLASS_MAP_PATH = Path(__file__).resolve().with_name("classes.json")

# Horizontal flip and scale jitter follow the official pipeline design.
AUG_RAND_HFLIP = True
# You may switch to 0.8, 1.25 if you want the milder SSJ variant.
AUG_SCALE_MIN = 0.1
AUG_SCALE_MAX = 2.0

# The official implementation removes occluded objects whose remaining visible
# area is less than this threshold.
COPY_PASTE_OCCLUDED_OBJ_THRESHOLD = 300

SAVE_METADATA = False
OVERWRITE_EXISTING = False
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    json_path: Path
    stem: str


@dataclass
class ShapeMask:
    label: str
    mask: np.ndarray
    group_id: Any = None
    flags: dict[str, Any] | None = None


@dataclass
class TransformInfo:
    flip_applied: bool
    random_scale: float
    scaled_size: tuple[int, int]
    resized_size: tuple[int, int]
    offset: tuple[int, int]
    desired_size: tuple[int, int]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def collect_dataset_items(images_folder: Path, annotations_folder: Path) -> list[DatasetItem]:
    if not images_folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {images_folder}")
    if not annotations_folder.is_dir():
        raise FileNotFoundError(f"Annotation folder not found: {annotations_folder}")

    items: list[DatasetItem] = []
    for image_path in sorted(images_folder.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        json_path = annotations_folder / f"{image_path.stem}.json"
        if not json_path.is_file():
            print(f"Skipping {image_path.name}: missing JSON")
            continue

        items.append(DatasetItem(image_path=image_path, json_path=json_path, stem=image_path.stem))

    if not items:
        raise RuntimeError(
            f"No valid image/JSON pairs found in:\n"
            f"  images: {images_folder}\n"
            f"  json:   {annotations_folder}"
        )

    return items


def load_class_map(path: Path) -> dict[str, int]:
    """Load the fixed class-name -> class-ID mapping from classes.json.

    The ``names`` list is authoritative: names[0] is class 0, etc.
    If ``label_to_id`` is also present, it must agree exactly.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared classes.json not found: {path}\n"
            "Place classes.json beside this script or update CLASS_MAP_PATH."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    names = data.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f'Invalid classes.json: "names" must be a non-empty list: {path}')

    clean_names: list[str] = []
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid class name at index {index}: {name!r}")
        clean_names.append(name.strip())

    if len(clean_names) != len(set(clean_names)):
        raise ValueError("Invalid classes.json: duplicate class names found.")

    label_to_id = {label: index for index, label in enumerate(clean_names)}

    stored_map = data.get("label_to_id")
    if stored_map is not None and stored_map != label_to_id:
        raise ValueError(
            'Invalid classes.json: "names" and "label_to_id" contain inconsistent mappings.'
        )

    return label_to_id


def validate_json_labels(
    data: dict[str, Any],
    label_to_id: dict[str, int],
    json_path: Path | None = None,
) -> None:
    """Fail fast if a polygon annotation uses a label outside classes.json."""
    unknown: set[str] = set()
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if not isinstance(label, str) or label not in label_to_id:
            unknown.add(str(label))

    if unknown:
        where = f" in {json_path}" if json_path is not None else ""
        raise ValueError(
            f"Unknown LabelMe label(s){where}: {sorted(unknown)}. "
            "All dataset classes must be defined in the shared classes.json."
        )


def valid_object_shapes(
    data: dict[str, Any],
    label_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Return valid polygon instances from ANY class in classes.json."""
    shapes: list[dict[str, Any]] = []
    for shape in data.get("shapes", []):
        if shape.get("label") not in label_to_id:
            continue
        if shape.get("shape_type", "polygon") not in {None, "polygon"}:
            continue

        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        if abs(cv2.contourArea(points)) < 1.0:
            continue
        shapes.append(shape)
    return shapes


def donor_candidates(
    items: list[DatasetItem],
    label_to_id: dict[str, int],
) -> list[DatasetItem]:
    candidates: list[DatasetItem] = []
    for item in items:
        try:
            data = load_json(item.json_path)
            validate_json_labels(data, label_to_id, item.json_path)
            if valid_object_shapes(data, label_to_id):
                candidates.append(item)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"Ignoring invalid donor JSON {item.json_path}: {error}")

    if not candidates:
        raise RuntimeError(
            "No donor JSON contains a valid polygon whose label is present in classes.json."
        )
    return candidates


def choose_donor(candidates: list[DatasetItem], rng: random.Random) -> DatasetItem:
    # Match the official pipeline more closely: donor and target are sampled
    # independently and may therefore be the same image.
    return rng.choice(candidates)


def polygon_to_mask(points: list[list[float]] | np.ndarray, height: int, width: int) -> np.ndarray:
    contour = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
    if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 2:
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    return mask


def shapes_to_shape_masks(
    data: dict[str, Any],
    height: int,
    width: int,
    label_to_id: dict[str, int],
) -> list[ShapeMask]:
    """Convert all mapped polygon classes to masks while preserving labels."""
    output: list[ShapeMask] = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in label_to_id:
            continue
        if shape.get("shape_type", "polygon") not in {None, "polygon"}:
            continue
        points = shape.get("points", [])
        mask = polygon_to_mask(points, height, width)
        if np.count_nonzero(mask) == 0:
            continue
        output.append(
            ShapeMask(
                label=label,
                mask=mask,
                group_id=shape.get("group_id"),
                flags=copy.deepcopy(shape.get("flags", {})),
            )
        )
    return output


def filter_shape_masks_by_classes(
    shapes: list[ShapeMask],
    label_to_id: dict[str, int],
) -> list[ShapeMask]:
    return [
        shape
        for shape in shapes
        if shape.label in label_to_id and np.count_nonzero(shape.mask) > 0
    ]


def sample_offset(max_offset: int, rng: random.Random) -> int:
    if max_offset <= 0:
        return 0
    # Mimics TF's int(max_offset * uniform[0, 1)).
    return int(rng.random() * max_offset)


def apply_preprocess_to_image_and_masks(
    image: np.ndarray,
    shapes: list[ShapeMask],
    desired_size: tuple[int, int],
    rng: random.Random,
) -> tuple[np.ndarray, list[ShapeMask], TransformInfo]:
    desired_h, desired_w = desired_size
    image_h, image_w = image.shape[:2]

    flip_applied = AUG_RAND_HFLIP and (rng.random() < 0.5)
    if flip_applied:
        image = cv2.flip(image, 1)
        flipped_shapes = []
        for shape in shapes:
            flipped_mask = cv2.flip(shape.mask, 1)
            flipped_shapes.append(
                ShapeMask(
                    label=shape.label,
                    mask=flipped_mask,
                    group_id=shape.group_id,
                    flags=copy.deepcopy(shape.flags) if shape.flags is not None else {},
                )
            )
        shapes = flipped_shapes

    random_jittering = (AUG_SCALE_MIN != 1.0 or AUG_SCALE_MAX != 1.0)
    if random_jittering:
        random_scale = rng.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
        scaled_target_h = int(round(random_scale * desired_h))
        scaled_target_w = int(round(random_scale * desired_w))
    else:
        random_scale = 1.0
        scaled_target_h = desired_h
        scaled_target_w = desired_w

    scale = min(scaled_target_h / image_h, scaled_target_w / image_w)
    resized_h = max(1, int(round(image_h * scale)))
    resized_w = max(1, int(round(image_w * scale)))

    max_offset_y = max(resized_h - desired_h, 0)
    max_offset_x = max(resized_w - desired_w, 0)
    if random_jittering:
        offset_y = sample_offset(max_offset_y, rng)
        offset_x = sample_offset(max_offset_x, rng)
    else:
        offset_y = 0
        offset_x = 0

    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    resized_shapes: list[ShapeMask] = []
    for shape in shapes:
        resized_mask = cv2.resize(shape.mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask > 127).astype(np.uint8) * 255
        resized_shapes.append(
            ShapeMask(
                label=shape.label,
                mask=resized_mask,
                group_id=shape.group_id,
                flags=copy.deepcopy(shape.flags) if shape.flags is not None else {},
            )
        )

    if random_jittering:
        cropped_image = resized_image[offset_y:offset_y + desired_h, offset_x:offset_x + desired_w]
    else:
        cropped_image = resized_image

    output_image = np.zeros((desired_h, desired_w, 3), dtype=resized_image.dtype)
    copy_h = min(desired_h, cropped_image.shape[0])
    copy_w = min(desired_w, cropped_image.shape[1])
    output_image[:copy_h, :copy_w] = cropped_image[:copy_h, :copy_w]

    transformed_shapes: list[ShapeMask] = []
    for shape in resized_shapes:
        if random_jittering:
            cropped_mask = shape.mask[offset_y:offset_y + desired_h, offset_x:offset_x + desired_w]
        else:
            cropped_mask = shape.mask

        output_mask = np.zeros((desired_h, desired_w), dtype=np.uint8)
        mask_h = min(desired_h, cropped_mask.shape[0])
        mask_w = min(desired_w, cropped_mask.shape[1])
        if mask_h > 0 and mask_w > 0:
            output_mask[:mask_h, :mask_w] = cropped_mask[:mask_h, :mask_w]

        output_mask = (output_mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(output_mask) == 0:
            continue
        transformed_shapes.append(
            ShapeMask(
                label=shape.label,
                mask=output_mask,
                group_id=shape.group_id,
                flags=copy.deepcopy(shape.flags) if shape.flags is not None else {},
            )
        )

    info = TransformInfo(
        flip_applied=flip_applied,
        random_scale=float(random_scale),
        scaled_size=(scaled_target_h, scaled_target_w),
        resized_size=(resized_h, resized_w),
        offset=(offset_y, offset_x),
        desired_size=desired_size,
    )
    return output_image, transformed_shapes, info


def select_random_nonempty_subset(
    donor_shapes: list[ShapeMask],
    rng: random.Random,
) -> list[ShapeMask]:
    # Official code samples m uniformly from 0..N. Here we use 1..N so every
    # offline-generated sample contains at least one pasted instance.
    count = rng.randint(1, len(donor_shapes))
    indices = list(range(len(donor_shapes)))
    rng.shuffle(indices)
    chosen = indices[:count]
    return [donor_shapes[idx] for idx in chosen]


def compose_from_union_mask(
    target_image: np.ndarray,
    donor_image: np.ndarray,
    donor_masks: list[ShapeMask],
) -> tuple[np.ndarray, np.ndarray]:
    union_mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
    for shape in donor_masks:
        union_mask = np.maximum(union_mask, (shape.mask > 0).astype(np.uint8) * 255)

    result = target_image.copy()
    result[union_mask > 0] = donor_image[union_mask > 0]
    return result, union_mask


def update_target_shapes_after_occlusion(
    target_shapes: list[ShapeMask],
    union_mask: np.ndarray,
) -> list[ShapeMask]:
    updated: list[ShapeMask] = []
    inverse_union = np.where(union_mask > 0, 0, 255).astype(np.uint8)

    for shape in target_shapes:
        original_mask = (shape.mask > 0).astype(np.uint8) * 255
        new_mask = cv2.bitwise_and(original_mask, inverse_union)

        original_area = int(np.count_nonzero(original_mask))
        new_area = int(np.count_nonzero(new_mask))
        was_occluded = new_area != original_area

        if new_area == 0:
            continue
        if was_occluded and new_area <= COPY_PASTE_OCCLUDED_OBJ_THRESHOLD:
            continue

        updated.append(
            ShapeMask(
                label=shape.label,
                mask=new_mask,
                group_id=shape.group_id,
                flags=copy.deepcopy(shape.flags) if shape.flags is not None else {},
            )
        )

    return updated


def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []

    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        contour = contour.reshape(-1, 2)
        if contour.shape[0] < 3:
            continue
        if abs(cv2.contourArea(contour.astype(np.float32))) < 1.0:
            continue
        polygon = [[float(x), float(y)] for x, y in contour]
        polygons.append(polygon)

    return polygons


def shape_masks_to_labelme_shapes(shapes: list[ShapeMask]) -> list[dict[str, Any]]:
    output_shapes: list[dict[str, Any]] = []
    for shape in shapes:
        polygons = mask_to_polygons(shape.mask)
        for polygon in polygons:
            output_shapes.append(
                {
                    "label": shape.label,
                    "points": polygon,
                    "group_id": shape.group_id,
                    "shape_type": "polygon",
                    "flags": copy.deepcopy(shape.flags) if shape.flags is not None else {},
                }
            )
    return output_shapes


def pick_transformed_donor_with_objects(
    donor_items: list[DatasetItem],
    desired_size: tuple[int, int],
    rng: random.Random,
    label_to_id: dict[str, int],
) -> tuple[DatasetItem, np.ndarray, list[ShapeMask], TransformInfo]:
    max_attempts = 50
    last_item: DatasetItem | None = None
    last_info: TransformInfo | None = None

    for _ in range(max_attempts):
        donor_item = choose_donor(donor_items, rng)
        last_item = donor_item

        donor_image = cv2.imread(str(donor_item.image_path), cv2.IMREAD_COLOR)
        if donor_image is None:
            continue

        donor_data = load_json(donor_item.json_path)
        validate_json_labels(donor_data, label_to_id, donor_item.json_path)
        donor_shape_masks = filter_shape_masks_by_classes(
            shapes_to_shape_masks(
                donor_data,
                donor_image.shape[0],
                donor_image.shape[1],
                label_to_id,
            ),
            label_to_id,
        )
        if not donor_shape_masks:
            continue

        transformed_donor_image, transformed_donor_shapes, donor_info = apply_preprocess_to_image_and_masks(
            donor_image,
            donor_shape_masks,
            desired_size=desired_size,
            rng=rng,
        )
        last_info = donor_info

        transformed_donor_shapes = [
            shape for shape in transformed_donor_shapes if np.count_nonzero(shape.mask) > 0
        ]
        if transformed_donor_shapes:
            return donor_item, transformed_donor_image, transformed_donor_shapes, donor_info

    raise RuntimeError(
        "Could not find a donor with at least one visible transformed mapped object "
        f"after {max_attempts} attempts. Last donor: {getattr(last_item, 'image_path', None)}, "
        f"last_transform: {last_info}"
    )


def generate_one(
    background_item: DatasetItem,
    donor_items: list[DatasetItem],
    label_to_id: dict[str, int],
) -> int:
    output_image_path = OUTPUT_IMAGES_FOLDER / f"{background_item.stem}.jpg"
    output_json_path = OUTPUT_ANNOTATIONS_FOLDER / f"{background_item.stem}.json"
    output_metadata_path = OUTPUT_METADATA_FOLDER / f"{background_item.stem}.json"

    if (
        not OVERWRITE_EXISTING
        and output_image_path.is_file()
        and output_json_path.is_file()
    ):
        print(f"Skipping existing output: {background_item.stem}")
        return 0

    rng = random.Random(stable_seed(MASTER_SEED, background_item.image_path.name, "simple_copy_paste_exact"))

    background = cv2.imread(str(background_item.image_path), cv2.IMREAD_COLOR)
    if background is None:
        raise RuntimeError(f"Could not read background image: {background_item.image_path}")

    background_data = load_json(background_item.json_path)
    validate_json_labels(background_data, label_to_id, background_item.json_path)
    desired_size = (background.shape[0], background.shape[1])
    background_shapes = shapes_to_shape_masks(
        background_data,
        background.shape[0],
        background.shape[1],
        label_to_id,
    )

    transformed_background, transformed_background_shapes, background_info = apply_preprocess_to_image_and_masks(
        background,
        background_shapes,
        desired_size=desired_size,
        rng=rng,
    )

    donor_item, transformed_donor_image, transformed_donor_shapes, donor_info = pick_transformed_donor_with_objects(
        donor_items,
        desired_size=desired_size,
        rng=rng,
        label_to_id=label_to_id,
    )

    selected_donor_shapes = select_random_nonempty_subset(transformed_donor_shapes, rng)
    composed_image, donor_union_mask = compose_from_union_mask(
        transformed_background,
        transformed_donor_image,
        selected_donor_shapes,
    )

    updated_target_shapes = update_target_shapes_after_occlusion(
        transformed_background_shapes,
        donor_union_mask,
    )
    final_shapes = updated_target_shapes + selected_donor_shapes

    output_data = {
        "version": background_data.get("version", "5.0.1"),
        "flags": copy.deepcopy(background_data.get("flags", {})),
        "shapes": shape_masks_to_labelme_shapes(final_shapes),
        "imagePath": output_image_path.name,
        "imageData": None,
        "imageHeight": int(desired_size[0]),
        "imageWidth": int(desired_size[1]),
    }

    if not cv2.imwrite(str(output_image_path), composed_image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write output image: {output_image_path}")
    save_json(output_json_path, output_data)

    if SAVE_METADATA:
        save_json(
            output_metadata_path,
            {
                "output_image": output_image_path.name,
                "background_image": background_item.image_path.name,
                "donor_image": donor_item.image_path.name,
                "requested_object_count": len(selected_donor_shapes),
                "pasted_object_count": len(selected_donor_shapes),
                "background_transform": {
                    "flip_applied": background_info.flip_applied,
                    "random_scale": background_info.random_scale,
                    "scaled_size": list(background_info.scaled_size),
                    "resized_size": list(background_info.resized_size),
                    "offset": list(background_info.offset),
                    "desired_size": list(background_info.desired_size),
                },
                "donor_transform": {
                    "flip_applied": donor_info.flip_applied,
                    "random_scale": donor_info.random_scale,
                    "scaled_size": list(donor_info.scaled_size),
                    "resized_size": list(donor_info.resized_size),
                    "offset": list(donor_info.offset),
                    "desired_size": list(donor_info.desired_size),
                },
                "objects": [
                    {
                        "label": shape.label,
                        "class_id": label_to_id[shape.label],
                        "visible_area": int(np.count_nonzero(shape.mask)),
                    }
                    for shape in selected_donor_shapes
                ],
            },
        )

    print(
        f"Generated {output_image_path.name}: donor={donor_item.image_path.name}, "
        f"pasted={len(selected_donor_shapes)}"
    )
    return 1


def main() -> None:
    label_to_id = load_class_map(CLASS_MAP_PATH)
    print(f"Loaded shared class map: {CLASS_MAP_PATH}")
    print(f"Number of classes: {len(label_to_id)}")

    OUTPUT_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANNOTATIONS_FOLDER.mkdir(parents=True, exist_ok=True)
    if SAVE_METADATA:
        OUTPUT_METADATA_FOLDER.mkdir(parents=True, exist_ok=True)

    background_items = collect_dataset_items(
        BACKGROUND_IMAGES_FOLDER,
        BACKGROUND_ANNOTATIONS_FOLDER,
    )
    object_items = collect_dataset_items(
        OBJECT_IMAGES_FOLDER,
        OBJECT_ANNOTATIONS_FOLDER,
    )
    donors = donor_candidates(object_items, label_to_id)

    generated = 0
    for background_item in background_items:
        generated += generate_one(background_item, donors, label_to_id)

    output_images = list(OUTPUT_IMAGES_FOLDER.glob("*.jpg"))
    output_jsons = list(OUTPUT_ANNOTATIONS_FOLDER.glob("*.json"))

    print("\nDone")
    print(f"Valid backgrounds: {len(background_items)}")
    print(f"Generated this run: {generated}")
    print(f"Total output images: {len(output_images)}")
    print(f"Total output JSONs: {len(output_jsons)}")
    print(f"Output root: {OUTPUT_ROOT}")

    if len(output_images) != len(output_jsons):
        print("WARNING: output image and JSON counts differ.")


if __name__ == "__main__":
    main()
