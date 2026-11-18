#!/usr/bin/env python3
"""
Offline MULTICLASS LabelMe wrapper for the authors' official ``instaboostfast``
package.

This version is intended for datasets such as D2S with many object classes.

Key behavior
------------
1. Reads ONE shared ``classes.json``.
2. Uses that file as the authoritative label -> class-ID mapping.
3. Sends all mapped polygon instances in an image to ONE InstaBoost call.
4. Preserves the original class label of every augmented instance.
5. Saves augmented image/LabelMe JSON pairs.
6. Refuses unknown labels instead of silently creating new IDs.

The InstaBoost algorithm itself is NOT reimplemented or changed.

Expected classes.json format
----------------------------
{
  "names": [
    "class_a",
    "class_b",
    "class_c"
  ],
  "label_to_id": {
    "class_a": 0,
    "class_b": 1,
    "class_c": 2
  }
}

The ``names`` list is authoritative:
    names[0] -> class ID 0
    names[1] -> class ID 1
    ...

If ``label_to_id`` is also present, it is checked for consistency.

Example
-------
python generate_instaboostfast_labelme_multiclass.py \
    --images "/path/train/images" \
    --json "/path/train/json" \
    --output "/path/output_instaboost" \
    --class-map "/path/classes.json" \
    --overwrite

If classes.json is in the same directory as this script, ``--class-map`` can
be omitted.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as cocomask

try:
    import instaboostfast as instaboost
except ImportError as exc:
    raise SystemExit(
        "instaboostfast is not installed. Run:\n"
        "  pip install instaboostfast==0.1.2\n"
        "and then rerun this script."
    ) from exc


EXPECTED_INSTABOOSTFAST_VERSION = "0.1.2"
VALID_IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
)

DEFAULT_CLASS_MAP = Path(__file__).resolve().with_name("classes.json")


# ---------------------------------------------------------------------------
# InstaBoost compatibility patch
# ---------------------------------------------------------------------------

def install_instaboostfast_numpy_compat_patch() -> None:
    """Patch one legacy NumPy-incompatible line in instaboostfast 0.1.2.

    The official implementation calls ``np.subtract(contours, 1)`` where
    ``contours`` is a list of arrays with different lengths. Modern NumPy
    refuses to convert that ragged list into one homogeneous ndarray.

    Applying the same subtraction separately to each contour is mathematically
    identical and does not change the InstaBoost algorithm, transformations,
    placement policy, masks, or randomness.
    """
    import instaboostfast.affine_transform as affine_transform
    import instaboostfast.pycococreater as pycococreater

    def binary_mask_to_polygon_compat(binary_mask, tolerance=0):
        polygons = []

        padded_binary_mask = np.pad(
            binary_mask,
            pad_width=1,
            mode="constant",
            constant_values=0,
        )

        contours = pycococreater.measure.find_contours(
            padded_binary_mask,
            0.5,
        )

        # Official source uses:
        #     contours = np.subtract(contours, 1)
        # This per-contour form is equivalent but supports ragged contour lists
        # under modern NumPy.
        contours = [
            np.subtract(contour, 1)
            for contour in contours
        ]

        for contour in contours:
            contour = pycococreater.close_contour(contour)
            contour = pycococreater.measure.approximate_polygon(
                contour,
                tolerance,
            )

            if len(contour) < 3:
                continue

            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            segmentation = [
                0 if coordinate < 0 else coordinate
                for coordinate in segmentation
            ]
            polygons.append(segmentation)

        return polygons

    # affine_transform imported this function directly, so patch both names.
    pycococreater.binary_mask_to_polygon = binary_mask_to_polygon_compat
    affine_transform.binary_mask_to_polygon = binary_mask_to_polygon_compat


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the official InstaBoostFast operator to every multiclass "
            "LabelMe image/JSON pair using one shared classes.json mapping."
        )
    )

    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--class-map",
        type=Path,
        default=DEFAULT_CLASS_MAP,
        help=(
            "Shared classes.json. Default: classes.json beside this script. "
            "Use the SAME file used for train/val/test/synthetic YOLO conversion."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--allow-other-instaboostfast-version",
        action="store_true",
        help=(
            "Allow an installed version other than 0.1.2. By default the "
            "script stops, so the official package version is pinned."
        ),
    )

    heatmap_group = parser.add_mutually_exclusive_group()
    heatmap_group.add_argument(
        "--heatmap",
        dest="heatmap",
        action="store_true",
        help="Use the official appearance-consistency heatmap placement.",
    )
    heatmap_group.add_argument(
        "--no-heatmap",
        dest="heatmap",
        action="store_false",
        help="Use official random local jittering instead of map guidance.",
    )
    parser.set_defaults(heatmap=True)

    # Defaults below are kept from the previous wrapper / official
    # instaboostfast-MMDetection configuration.
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--dx", type=float, default=15)
    parser.add_argument("--dy", type=float, default=15)
    parser.add_argument("--theta-min", type=int, default=-1)
    parser.add_argument("--theta-max", type=int, default=1)
    parser.add_argument("--color-prob", type=float, default=0.5)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared class mapping
# ---------------------------------------------------------------------------

def load_class_map(path: Path) -> OrderedDict[str, int]:
    """Load the fixed label -> ID mapping from classes.json."""

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find classes.json:\n  {path}\n"
            "Pass it with --class-map, or place classes.json beside this script."
        )

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"classes.json must contain a JSON object: {path}")

    names = data.get("names")

    if not isinstance(names, list) or not names:
        raise ValueError(
            f'Invalid classes.json: "names" must be a non-empty list: {path}'
        )

    cleaned_names: list[str] = []

    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Invalid class name at index {index} in {path}: {name!r}"
            )
        cleaned_names.append(name.strip())

    if len(cleaned_names) != len(set(cleaned_names)):
        raise ValueError(f"Duplicate class names found in {path}")

    label_to_id: OrderedDict[str, int] = OrderedDict(
        (label, idx)
        for idx, label in enumerate(cleaned_names)
    )

    # If the redundant label_to_id field exists, verify it exactly.
    stored_map = data.get("label_to_id")
    if stored_map is not None:
        if stored_map != dict(label_to_id):
            raise ValueError(
                'Invalid classes.json: "names" and "label_to_id" describe '
                "different class IDs."
            )

    return label_to_id


def inverse_class_map(
    label_to_id: OrderedDict[str, int],
) -> dict[int, str]:
    return {class_id: label for label, class_id in label_to_id.items()}


# ---------------------------------------------------------------------------
# LabelMe helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("shapes", []), list):
        raise ValueError(f"Invalid LabelMe JSON: {path}")

    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def resolve_image(
    json_path: Path,
    data: dict[str, Any],
    images_root: Path,
    json_root: Path,
) -> Path:
    """Find the image corresponding to one LabelMe JSON."""

    relative_json_parent = json_path.relative_to(json_root).parent
    image_path_value = data.get("imagePath")

    candidates: list[Path] = []

    if isinstance(image_path_value, str) and image_path_value.strip():
        image_name = Path(image_path_value).name

        candidates.extend(
            [
                images_root / relative_json_parent / image_name,
                images_root / image_name,
                json_path.parent / image_path_value,
            ]
        )

    for extension in VALID_IMAGE_EXTENSIONS:
        candidates.extend(
            [
                images_root / relative_json_parent / f"{json_path.stem}{extension}",
                images_root / f"{json_path.stem}{extension}",
            ]
        )

    unique_existing: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate not in seen and candidate.is_file():
            unique_existing.append(candidate)
            seen.add(candidate)

    if len(unique_existing) == 1:
        return unique_existing[0]

    if not unique_existing:
        raise FileNotFoundError(f"No image found for {json_path}")

    raise RuntimeError(
        f"Ambiguous image match for {json_path}: "
        + ", ".join(str(path) for path in unique_existing)
    )


def clean_polygon(
    points: Any,
    width: int,
    height: int,
) -> list[list[float]]:
    """Clean and clip one polygon to the actual image dimensions."""

    if not isinstance(points, list):
        return []

    cleaned: list[list[float]] = []

    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        x = float(point[0])
        y = float(point[1])

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        x = float(np.clip(x, 0, max(width - 1, 0)))
        y = float(np.clip(y, 0, max(height - 1, 0)))

        if not cleaned or cleaned[-1] != [x, y]:
            cleaned.append([x, y])

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    if len(cleaned) < 3:
        return []

    contour = np.asarray(cleaned, dtype=np.float32)

    if abs(float(cv2.contourArea(contour))) < 1.0:
        return []

    return cleaned


def flatten_polygon(points: list[list[float]]) -> list[float]:
    return [coordinate for point in points for coordinate in point]


def validate_and_group_shapes(
    shapes: list[dict[str, Any]],
    label_to_id: OrderedDict[str, int],
) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """
    Validate the current LabelMe shapes and group polygons into instances.

    Every label in the input JSON must exist in classes.json.

    A LabelMe ``group_id`` may be shared by multiple polygon parts belonging to
    one instance. To avoid accidentally joining polygons from different
    classes, the grouping key includes BOTH class label and group_id.

    Shapes whose labels are in classes.json are augmentation targets.
    Non-polygon target shapes are rejected because this wrapper is for instance
    segmentation polygons.
    """

    grouped: OrderedDict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = OrderedDict()

    untouched: list[dict[str, Any]] = []
    anonymous_index = 0

    for shape_index, shape in enumerate(shapes):
        label = shape.get("label")

        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Shape {shape_index} has an invalid/missing label: {label!r}"
            )

        label = label.strip()

        if label not in label_to_id:
            raise ValueError(
                f"Label {label!r} is not present in the shared classes.json."
            )

        shape_type = shape.get("shape_type", "polygon")

        if shape_type != "polygon":
            raise ValueError(
                f"Class {label!r} contains unsupported shape_type "
                f"{shape_type!r}. This InstaBoost wrapper requires polygons."
            )

        group_id = shape.get("group_id")

        if group_id is None:
            key = (label, "anonymous", str(anonymous_index))
            anonymous_index += 1
        else:
            key = (
                label,
                "group",
                json.dumps(group_id, sort_keys=True),
            )

        grouped.setdefault(key, []).append(copy.deepcopy(shape))

    grouped_instances = [
        (key[0], instance_shapes)
        for key, instance_shapes in grouped.items()
    ]

    return grouped_instances, untouched


# ---------------------------------------------------------------------------
# LabelMe -> COCO-style annotations for InstaBoost
# ---------------------------------------------------------------------------

def coco_bbox_and_area(
    segmentation: list[list[float]],
    height: int,
    width: int,
) -> tuple[list[float], float]:
    rles = cocomask.frPyObjects(segmentation, height, width)
    merged = cocomask.merge(rles)

    bbox = [
        float(value)
        for value in cocomask.toBbox(merged).tolist()
    ]
    area = float(cocomask.area(merged))

    return bbox, area


def labelme_to_coco(
    data: dict[str, Any],
    label_to_id: OrderedDict[str, int],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Convert ALL mapped LabelMe polygon instances to InstaBoost annotations.

    category_id is the exact fixed 0-based ID from classes.json.
    This matches the way MMDetection passes class labels to InstaBoost.
    """

    groups, untouched = validate_and_group_shapes(
        data.get("shapes", []),
        label_to_id,
    )

    annotations: list[dict[str, Any]] = []

    for instance_index, (label, shapes) in enumerate(groups, start=1):
        segmentation: list[list[float]] = []
        templates: list[dict[str, Any]] = []

        for shape in shapes:
            points = clean_polygon(
                shape.get("points", []),
                width,
                height,
            )

            if not points:
                continue

            segmentation.append(flatten_polygon(points))
            templates.append(shape)

        if not segmentation:
            continue

        bbox, area = coco_bbox_and_area(
            segmentation,
            height,
            width,
        )

        shared_class_id = label_to_id[label]

        annotations.append(
            {
                "id": instance_index,
                "image_id": 1,

                # IMPORTANT:
                # This is the exact class ID from shared classes.json.
                "category_id": shared_class_id,

                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,

                # Metadata used only to restore LabelMe faithfully.
                # The previous wrapper already relies on InstaBoost's
                # deepcopy/update behavior preserving unknown keys.
                "_labelme_label": label,
                "_shared_class_id": shared_class_id,
                "_labelme_group_id": shapes[0].get("group_id"),
                "_labelme_templates": templates,
                "_labelme_instance_index": instance_index,
            }
        )

    if not annotations:
        raise ValueError(
            "No valid mapped polygon instances were found in this LabelMe JSON."
        )

    return annotations, untouched


# ---------------------------------------------------------------------------
# COCO-style InstaBoost output -> LabelMe
# ---------------------------------------------------------------------------

def segmentation_to_points(
    segmentation: Any,
    width: int,
    height: int,
) -> list[list[float]]:
    if not isinstance(segmentation, list) or len(segmentation) < 6:
        return []

    if len(segmentation) % 2 != 0:
        return []

    array = np.asarray(
        segmentation,
        dtype=np.float64,
    ).reshape(-1, 2)

    array[:, 0] = np.clip(
        array[:, 0],
        0,
        max(width - 1, 0),
    )
    array[:, 1] = np.clip(
        array[:, 1],
        0,
        max(height - 1, 0),
    )

    return clean_polygon(
        array.tolist(),
        width,
        height,
    )


def next_available_group_id(
    shapes: list[dict[str, Any]],
) -> int:
    numeric_ids = [
        int(shape["group_id"])
        for shape in shapes
        if isinstance(shape.get("group_id"), int)
    ]

    return max(numeric_ids, default=0) + 1


def recover_annotation_label(
    annotation: dict[str, Any],
    id_to_label: dict[int, str],
) -> str:
    """
    Recover the original class robustly.

    Priority:
    1. _labelme_label metadata
    2. original LabelMe template label
    3. category_id -> classes.json lookup
    """

    stored_label = annotation.get("_labelme_label")

    if isinstance(stored_label, str) and stored_label:
        return stored_label

    templates = annotation.get("_labelme_templates", [])

    if templates:
        template_label = templates[0].get("label")

        if isinstance(template_label, str) and template_label:
            return template_label

    category_id = annotation.get("category_id")

    if isinstance(category_id, np.integer):
        category_id = int(category_id)

    if isinstance(category_id, int) and category_id in id_to_label:
        return id_to_label[category_id]

    raise ValueError(
        "Could not recover class label from an InstaBoost annotation. "
        f"category_id={category_id!r}"
    )


def coco_to_labelme_shapes(
    annotations: list[dict[str, Any]],
    id_to_label: dict[int, str],
    untouched_shapes: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """
    Convert augmented InstaBoost polygons back to LabelMe while preserving
    each annotation's original class.
    """

    output_shapes = copy.deepcopy(untouched_shapes)
    synthetic_group_id = next_available_group_id(output_shapes)

    for annotation in annotations:
        label = recover_annotation_label(
            annotation,
            id_to_label,
        )

        segmentation = annotation.get("segmentation", [])

        if not isinstance(segmentation, list):
            raise ValueError(
                "The wrapper supports polygon output, not RLE output."
            )

        polygons = [
            segmentation_to_points(
                flat,
                width,
                height,
            )
            for flat in segmentation
        ]

        polygons = [
            points
            for points in polygons
            if points
        ]

        if not polygons:
            continue

        templates = annotation.get(
            "_labelme_templates",
            [],
        )
        original_group_id = annotation.get(
            "_labelme_group_id"
        )

        output_group_id = original_group_id

        # If one original instance is returned as multiple polygons and there
        # was no original group_id, give those polygon parts one new common ID.
        if output_group_id is None and len(polygons) > 1:
            output_group_id = synthetic_group_id
            synthetic_group_id += 1

        for polygon_index, points in enumerate(polygons):
            if templates:
                template = copy.deepcopy(
                    templates[
                        min(
                            polygon_index,
                            len(templates) - 1,
                        )
                    ]
                )
            else:
                template = {}

            # IMPORTANT: restore THIS instance's original class.
            template["label"] = label
            template["points"] = points
            template["group_id"] = output_group_id
            template["shape_type"] = "polygon"
            template.setdefault("flags", {})

            output_shapes.append(template)

    return output_shapes


# ---------------------------------------------------------------------------
# Diagnostics / reproducibility
# ---------------------------------------------------------------------------

def changed_pixel_fraction(
    before: np.ndarray,
    after: np.ndarray,
) -> float:
    if before.shape != after.shape:
        return 1.0

    if before.ndim == 2:
        changed = before != after
    else:
        changed = np.any(
            before != after,
            axis=2,
        )

    return float(changed.mean())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def build_config(args: argparse.Namespace) -> Any:
    """
    Keep the same InstaBoost configuration behavior as the previous wrapper.
    """

    return instaboost.InstaBoostConfig(
        action_candidate=("normal", "horizontal", "skip"),
        action_prob=(1, 0, 0),
        scale=(args.scale_min, args.scale_max),
        dx=args.dx,
        dy=args.dy,
        theta=(args.theta_min, args.theta_max),
        color_prob=args.color_prob,
        heatmap_flag=args.heatmap,
    )


def verify_package(
    args: argparse.Namespace,
) -> tuple[str, Path]:
    version = importlib.metadata.version(
        "instaboostfast"
    )
    package_path = Path(
        instaboost.__file__
    ).resolve()

    if (
        version != EXPECTED_INSTABOOSTFAST_VERSION
        and not args.allow_other_instaboostfast_version
    ):
        raise SystemExit(
            f"Expected instaboostfast=="
            f"{EXPECTED_INSTABOOSTFAST_VERSION}, "
            f"but found {version} at {package_path}."
        )

    return version, package_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    label_to_id = load_class_map(args.class_map)
    id_to_label = inverse_class_map(label_to_id)

    version, package_path = verify_package(args)
    install_instaboostfast_numpy_compat_patch()

    if args.scale_min > args.scale_max:
        raise SystemExit(
            "--scale-min cannot exceed --scale-max."
        )

    if args.theta_min > args.theta_max:
        raise SystemExit(
            "--theta-min cannot exceed --theta-max."
        )

    if not 0.0 <= args.color_prob <= 1.0:
        raise SystemExit(
            "--color-prob must be in [0, 1]."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_images = args.output / "images"
    output_json = args.output / "json"

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_json.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = build_config(args)

    json_files = sorted(
        args.json.rglob("*.json")
    )

    if not json_files:
        raise SystemExit(
            f"No JSON files found under {args.json}"
        )

    manifest_rows: list[dict[str, Any]] = []

    print(f"Official package: instaboostfast=={version}")
    print(f"Imported from:    {package_path}")
    print(f"Class map:        {args.class_map.resolve()}")
    print(f"Number of classes:{len(label_to_id):>5}")
    print(f"Heatmap guided:   {args.heatmap}")
    print("NumPy fix:         enabled (per-contour subtraction only)")
    print(f"Input pairs:       {len(json_files)}")
    print("Target classes:    ALL labels in classes.json")
    print("Apply gate:        none (official operator called for every image)")

    for index, json_path in enumerate(
        json_files,
        start=1,
    ):
        data = load_json(json_path)

        image_path = resolve_image(
            json_path,
            data,
            args.images,
            args.json,
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"OpenCV failed to read {image_path}"
            )

        height, width = image.shape[:2]

        # ALL mapped classes are included together in ONE annotation list.
        input_annotations, untouched_shapes = labelme_to_coco(
            data,
            label_to_id,
            width,
            height,
        )

        # This is the only augmentation call.
        # It invokes the authors' package directly; no InstaBoost operation
        # is copied or reimplemented here.
        new_annotations, new_image = instaboost.get_new_data(
            ori_anns=input_annotations,
            ori_img=image,
            config=config,
            background=None,
        )

        if not isinstance(
            new_image,
            np.ndarray,
        ):
            raise TypeError(
                "instaboostfast did not return a NumPy image."
            )

        if new_image.shape != image.shape:
            raise ValueError(
                f"Image shape changed from "
                f"{image.shape} to {new_image.shape}."
            )

        relative_json = json_path.relative_to(
            args.json
        )
        relative_parent = relative_json.parent

        output_image_path = (
            output_images
            / relative_parent
            / image_path.name
        )
        output_json_path = (
            output_json
            / relative_json
        )

        output_image_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not args.overwrite and (
            output_image_path.exists()
            or output_json_path.exists()
        ):
            raise FileExistsError(
                f"Output exists for {json_path.stem}; "
                "use --overwrite."
            )

        output_shapes = coco_to_labelme_shapes(
            new_annotations,
            id_to_label,
            untouched_shapes,
            width,
            height,
        )

        output_data = copy.deepcopy(data)
        output_data["shapes"] = output_shapes
        output_data["imagePath"] = output_image_path.name
        output_data["imageData"] = None
        output_data["imageHeight"] = int(height)
        output_data["imageWidth"] = int(width)

        if not cv2.imwrite(
            str(output_image_path),
            new_image,
        ):
            raise IOError(
                f"Could not write {output_image_path}"
            )

        save_json(
            output_json_path,
            output_data,
        )

        changed_fraction = changed_pixel_fraction(
            image,
            new_image,
        )

        input_classes = sorted(
            {
                ann["category_id"]
                for ann in input_annotations
            }
        )
        output_classes = sorted(
            {
                int(ann["category_id"])
                for ann in new_annotations
                if "category_id" in ann
            }
        )

        manifest_rows.append(
            {
                "json": str(relative_json),
                "image": image_path.name,
                "input_instances": len(input_annotations),
                "output_instances": len(new_annotations),
                "dropped_instances": (
                    len(input_annotations)
                    - len(new_annotations)
                ),
                "input_class_ids": ",".join(
                    str(x)
                    for x in input_classes
                ),
                "output_class_ids": ",".join(
                    str(x)
                    for x in output_classes
                ),
                "changed_pixel_fraction": (
                    f"{changed_fraction:.8f}"
                ),
                "unchanged_image": int(
                    changed_fraction == 0.0
                ),
            }
        )

        print(
            f"[{index:>5}/{len(json_files)}] "
            f"{relative_json}: "
            f"{len(input_annotations)} -> "
            f"{len(new_annotations)} instances, "
            f"classes={input_classes}, "
            f"changed={changed_fraction:.4%}"
        )

    manifest_path = args.output / "manifest.csv"

    fieldnames = [
        "json",
        "image",
        "input_instances",
        "output_instances",
        "dropped_instances",
        "input_class_ids",
        "output_class_ids",
        "changed_pixel_fraction",
        "unchanged_image",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "wrapper": (
            "Multiclass LabelMe <-> COCO conversion around "
            "official instaboostfast"
        ),
        "official_package": {
            "name": "instaboostfast",
            "version": version,
            "import_path": str(package_path),
            "package_init_sha256": file_sha256(
                package_path
            ),
        },
        "operator_call": (
            "instaboostfast.get_new_data"
            "(..., background=None)"
        ),
        "compatibility_patch": (
            "Legacy np.subtract(contours, 1) replaced "
            "at runtime by per-contour "
            "np.subtract(contour, 1); algorithm unchanged"
        ),
        "multiclass": True,
        "class_map": {
            "path": str(args.class_map.resolve()),
            "num_classes": len(label_to_id),
            "names": list(label_to_id.keys()),
            "label_to_id": dict(label_to_id),
        },
        "all_mapped_classes_processed_together_per_image": True,
        "augmentation_applied_to_every_input_image": True,
        "seed": args.seed,
        "config": {
            "action_candidate": [
                "normal",
                "horizontal",
                "skip",
            ],
            "action_prob": [1, 0, 0],
            "scale": [
                args.scale_min,
                args.scale_max,
            ],
            "dx": args.dx,
            "dy": args.dy,
            "theta_degrees": [
                args.theta_min,
                args.theta_max,
            ],
            "color_prob": args.color_prob,
            "heatmap_flag": args.heatmap,
        },
        "input_pairs": len(json_files),
        "unchanged_images": sum(
            row["unchanged_image"]
            for row in manifest_rows
        ),
        "total_input_instances": sum(
            row["input_instances"]
            for row in manifest_rows
        ),
        "total_output_instances": sum(
            row["output_instances"]
            for row in manifest_rows
        ),
    }

    save_json(
        args.output / "generation_summary.json",
        summary,
    )

    print(f"\nDone. Images:   {output_images}")
    print(f"JSON:     {output_json}")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
