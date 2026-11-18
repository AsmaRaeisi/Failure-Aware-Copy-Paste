#python labelme2coco_conv.py \
#  --classes /path/to/classes.json \
#  --labelme-dir /path/to/split/json \
#  --export-dir /path/to/output


import argparse
import json
from pathlib import Path

import labelme2coco


LABELME_DIR = Path(
    "**/json"
)

EXPORT_DIR = Path(
    "**/coco_style"
)

CLASSES_JSON = Path("**/classes.json")

def main():
    parser = argparse.ArgumentParser(
        description="Convert LabelMe annotations using fixed class IDs."
    )
    parser.add_argument(
        "--classes", type=Path,
        default=CLASSES_JSON,
        help="JSON file containing a label_to_id mapping (default: CLASSES_JSON).",
    )
    parser.add_argument("--labelme-dir", type=Path, default=LABELME_DIR)
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    args = parser.parse_args()

    with args.classes.open("r", encoding="utf-8") as mapping_file:
        classes = json.load(mapping_file)
    label_to_id = classes.get("label_to_id") if isinstance(classes, dict) else None
    if not isinstance(label_to_id, dict) or not label_to_id:
        parser.error("classes JSON must contain a non-empty label_to_id object.")
    if any(
        not isinstance(label, str) or not label
        or type(category_id) is not int or category_id < 0
        for label, category_id in label_to_id.items()
    ):
        parser.error("label_to_id must map non-empty labels to non-negative integer IDs.")
    if len(set(label_to_id.values())) != len(label_to_id):
        parser.error("Each label must have a unique category ID.")

    categories = [
        {"id": category_id, "name": label}
        for label, category_id in sorted(label_to_id.items(), key=lambda item: item[1])
    ]
    coco = labelme2coco.get_coco_from_labelme_folder(
        str(args.labelme_dir), coco_category_list=categories
    ).json
    converted_mapping = {
        category["name"]: category["id"] for category in coco["categories"]
    }
    if converted_mapping != label_to_id:
        unknown_labels = sorted(set(converted_mapping) - set(label_to_id))
        parser.error(
            "Converted categories do not match classes JSON. "
            f"Unknown labels: {unknown_labels}. Output was not written."
        )
    coco["categories"] = categories

    for image in coco["images"]:
        image["file_name"] = Path(image["file_name"]).name

    args.export_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.export_dir / "dataset.json"
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(coco, output_file)

    print(f"Created: {output_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Classes: {len(categories)}")
    if coco["images"]:
        print("Example file_name:", coco["images"][0]["file_name"])


if __name__ == "__main__":
    main()