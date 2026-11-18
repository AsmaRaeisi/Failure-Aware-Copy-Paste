#/proj/dcdl/users/x_asmra/envs/mmdet/bin/python labelme2coco_conv.py
import json
from pathlib import Path

import labelme2coco


LABELME_DIR = Path(
    "**/json"
)

EXPORT_DIR = Path(
    "**/coco_style"
)

OUTPUT_JSON = EXPORT_DIR / "dataset.json"


# Convert all LabelMe JSON files to COCO
labelme2coco.convert(
    str(LABELME_DIR),
    str(EXPORT_DIR),
    train_split_rate=1.0,
)


# Rewrite image paths to filenames only
with OUTPUT_JSON.open("r", encoding="utf-8") as f:
    coco = json.load(f)

for image in coco["images"]:
    image["file_name"] = Path(image["file_name"]).name

with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(coco, f)

print(f"Created: {OUTPUT_JSON}")
print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print("Example file_name:", coco["images"][0]["file_name"])