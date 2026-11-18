#!/usr/bin/env python3
"""Offline LabelMe wrapper for the authors' official ``instaboostfast`` package.

The InstaBoost algorithm is not reimplemented here. This script only:
1. reads LabelMe polygons,
2. converts them to COCO-style annotation dictionaries,
3. calls ``instaboostfast.get_new_data`` unchanged,
4. converts the returned COCO polygons back to LabelMe JSON,
5. saves one output image/JSON pair for every input pair.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the official InstaBoostFast operator to every LabelMe "
            "image/JSON pair and save offline augmented pairs."
        )
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Object")
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

    # Defaults below are the official instaboostfast/MMDetection defaults.
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--dx", type=float, default=15)
    parser.add_argument("--dy", type=float, default=15)
    parser.add_argument("--theta-min", type=int, default=-1)
    parser.add_argument("--theta-max", type=int, default=1)
    parser.add_argument("--color-prob", type=float, default=0.5)
    return parser.parse_args()


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


def clean_polygon(points: Any, width: int, height: int) -> list[list[float]]:
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


def group_target_shapes(
    shapes: list[dict[str, Any]],
    target_label: str,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    untouched: list[dict[str, Any]] = []
    anonymous_index = 0

    for shape in shapes:
        if shape.get("label") != target_label:
            untouched.append(copy.deepcopy(shape))
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(
                f"Target shape must be a polygon, got {shape.get('shape_type')!r}."
            )

        group_id = shape.get("group_id")
        if group_id is None:
            key = ("anonymous", str(anonymous_index))
            anonymous_index += 1
        else:
            key = ("group", json.dumps(group_id, sort_keys=True))
        grouped.setdefault(key, []).append(copy.deepcopy(shape))

    return list(grouped.values()), untouched


def coco_bbox_and_area(
    segmentation: list[list[float]],
    height: int,
    width: int,
) -> tuple[list[float], float]:
    rles = cocomask.frPyObjects(segmentation, height, width)
    merged = cocomask.merge(rles)
    bbox = [float(value) for value in cocomask.toBbox(merged).tolist()]
    area = float(cocomask.area(merged))
    return bbox, area


def labelme_to_coco(
    data: dict[str, Any],
    target_label: str,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups, untouched = group_target_shapes(data.get("shapes", []), target_label)
    annotations: list[dict[str, Any]] = []

    for instance_index, shapes in enumerate(groups, start=1):
        segmentation: list[list[float]] = []
        templates: list[dict[str, Any]] = []

        for shape in shapes:
            points = clean_polygon(shape.get("points", []), width, height)
            if not points:
                continue
            segmentation.append(flatten_polygon(points))
            templates.append(shape)

        if not segmentation:
            continue

        bbox, area = coco_bbox_and_area(segmentation, height, width)
        annotations.append(
            {
                "id": instance_index,
                "image_id": 1,
                "category_id": 1,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                # Unknown keys are preserved by the official package's
                # deepcopy/update path and are used only for LabelMe export.
                "_labelme_group_id": shapes[0].get("group_id"),
                "_labelme_templates": templates,
                "_labelme_instance_index": instance_index,
            }
        )

    if not annotations:
        raise ValueError(f"No valid {target_label!r} polygons were found.")
    return annotations, untouched


def segmentation_to_points(
    segmentation: Any,
    width: int,
    height: int,
) -> list[list[float]]:
    if not isinstance(segmentation, list) or len(segmentation) < 6:
        return []
    if len(segmentation) % 2 != 0:
        return []
    array = np.asarray(segmentation, dtype=np.float64).reshape(-1, 2)
    array[:, 0] = np.clip(array[:, 0], 0, max(width - 1, 0))
    array[:, 1] = np.clip(array[:, 1], 0, max(height - 1, 0))
    return clean_polygon(array.tolist(), width, height)


def next_available_group_id(shapes: list[dict[str, Any]]) -> int:
    numeric_ids = [
        int(shape["group_id"])
        for shape in shapes
        if isinstance(shape.get("group_id"), int)
    ]
    return max(numeric_ids, default=0) + 1


def coco_to_labelme_shapes(
    annotations: list[dict[str, Any]],
    target_label: str,
    untouched_shapes: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    output_shapes = copy.deepcopy(untouched_shapes)
    synthetic_group_id = next_available_group_id(output_shapes)

    for annotation in annotations:
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            raise ValueError("The wrapper supports polygon output, not RLE output.")

        polygons = [
            segmentation_to_points(flat, width, height)
            for flat in segmentation
        ]
        polygons = [points for points in polygons if points]
        if not polygons:
            continue

        templates = annotation.get("_labelme_templates", [])
        original_group_id = annotation.get("_labelme_group_id")
        output_group_id = original_group_id
        if output_group_id is None and len(polygons) > 1:
            output_group_id = synthetic_group_id
            synthetic_group_id += 1

        for polygon_index, points in enumerate(polygons):
            if templates:
                template = copy.deepcopy(
                    templates[min(polygon_index, len(templates) - 1)]
                )
            else:
                template = {}

            template["label"] = target_label
            template["points"] = points
            template["group_id"] = output_group_id
            template["shape_type"] = "polygon"
            template.setdefault("flags", {})
            output_shapes.append(template)

    return output_shapes


def changed_pixel_fraction(before: np.ndarray, after: np.ndarray) -> float:
    if before.shape != after.shape:
        return 1.0
    if before.ndim == 2:
        changed = before != after
    else:
        changed = np.any(before != after, axis=2)
    return float(changed.mean())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(args: argparse.Namespace) -> Any:
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


def verify_package(args: argparse.Namespace) -> tuple[str, Path]:
    version = importlib.metadata.version("instaboostfast")
    package_path = Path(instaboost.__file__).resolve()
    if (
        version != EXPECTED_INSTABOOSTFAST_VERSION
        and not args.allow_other_instaboostfast_version
    ):
        raise SystemExit(
            f"Expected instaboostfast=={EXPECTED_INSTABOOSTFAST_VERSION}, "
            f"but found {version} at {package_path}."
        )
    return version, package_path


def main() -> int:
    args = parse_args()
    version, package_path = verify_package(args)
    install_instaboostfast_numpy_compat_patch()

    if args.scale_min > args.scale_max:
        raise SystemExit("--scale-min cannot exceed --scale-max.")
    if args.theta_min > args.theta_max:
        raise SystemExit("--theta-min cannot exceed --theta-max.")
    if not 0.0 <= args.color_prob <= 1.0:
        raise SystemExit("--color-prob must be in [0, 1].")

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_images = args.output / "images"
    output_json = args.output / "json"
    output_images.mkdir(parents=True, exist_ok=True)
    output_json.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    json_files = sorted(args.json.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found under {args.json}")

    manifest_rows: list[dict[str, Any]] = []

    print(f"Official package: instaboostfast=={version}")
    print(f"Imported from:    {package_path}")
    print(f"Heatmap guided:  {args.heatmap}")
    print("NumPy fix:        enabled (per-contour subtraction only)")
    print(f"Input pairs:      {len(json_files)}")
    print("Apply gate:       none (official operator called for every image)")

    for index, json_path in enumerate(json_files, start=1):
        data = load_json(json_path)
        image_path = resolve_image(json_path, data, args.images, args.json)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to read {image_path}")

        height, width = image.shape[:2]
        input_annotations, untouched_shapes = labelme_to_coco(
            data, args.label, width, height
        )

        # This is the only augmentation call. It invokes the authors' package
        # directly; no InstaBoost operation is copied or reimplemented here.
        new_annotations, new_image = instaboost.get_new_data(
            ori_anns=input_annotations,
            ori_img=image,
            config=config,
            background=None,
        )

        if not isinstance(new_image, np.ndarray):
            raise TypeError("instaboostfast did not return a NumPy image.")
        if new_image.shape != image.shape:
            raise ValueError(
                f"Image shape changed from {image.shape} to {new_image.shape}."
            )

        relative_json = json_path.relative_to(args.json)
        relative_parent = relative_json.parent
        output_image_path = output_images / relative_parent / image_path.name
        output_json_path = output_json / relative_json
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.overwrite and (
            output_image_path.exists() or output_json_path.exists()
        ):
            raise FileExistsError(
                f"Output exists for {json_path.stem}; use --overwrite."
            )

        output_shapes = coco_to_labelme_shapes(
            new_annotations,
            args.label,
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

        if not cv2.imwrite(str(output_image_path), new_image):
            raise IOError(f"Could not write {output_image_path}")
        save_json(output_json_path, output_data)

        changed_fraction = changed_pixel_fraction(image, new_image)
        manifest_rows.append(
            {
                "json": str(relative_json),
                "image": image_path.name,
                "input_instances": len(input_annotations),
                "output_instances": len(new_annotations),
                "dropped_instances": len(input_annotations) - len(new_annotations),
                "changed_pixel_fraction": f"{changed_fraction:.8f}",
                "unchanged_image": int(changed_fraction == 0.0),
            }
        )
        print(
            f"[{index:>5}/{len(json_files)}] {relative_json}: "
            f"{len(input_annotations)} -> {len(new_annotations)} instances, "
            f"changed={changed_fraction:.4%}"
        )

    manifest_path = args.output / "manifest.csv"
    fieldnames = [
        "json",
        "image",
        "input_instances",
        "output_instances",
        "dropped_instances",
        "changed_pixel_fraction",
        "unchanged_image",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "wrapper": "LabelMe <-> COCO conversion around official instaboostfast",
        "official_package": {
            "name": "instaboostfast",
            "version": version,
            "import_path": str(package_path),
            "package_init_sha256": file_sha256(package_path),
        },
        "operator_call": "instaboostfast.get_new_data(..., background=None)",
        "compatibility_patch": (
            "Legacy np.subtract(contours, 1) replaced at runtime by "
            "per-contour np.subtract(contour, 1); algorithm unchanged"
        ),
        "augmentation_applied_to_every_input_image": True,
        "seed": args.seed,
        "config": {
            "action_candidate": ["normal", "horizontal", "skip"],
            "action_prob": [1, 0, 0],
            "scale": [args.scale_min, args.scale_max],
            "dx": args.dx,
            "dy": args.dy,
            "theta_degrees": [args.theta_min, args.theta_max],
            "color_prob": args.color_prob,
            "heatmap_flag": args.heatmap,
        },
        "input_pairs": len(json_files),
        "unchanged_images": sum(row["unchanged_image"] for row in manifest_rows),
        "total_input_instances": sum(row["input_instances"] for row in manifest_rows),
        "total_output_instances": sum(row["output_instances"] for row in manifest_rows),
    }
    save_json(args.output / "generation_summary.json", summary)

    print(f"\nDone. Images:   {output_images}")
    print(f"JSON:     {output_json}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
