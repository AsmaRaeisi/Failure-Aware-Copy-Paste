"""
Out-of-fold (OOF) failure mining for copy-paste augmentation (Ultralytics YOLO-seg).

Each training image is analysed by a model that never saw it: the training set is
split into K folds, a model is trained on K-1 folds with the baseline recipe, and
failures are mined on the held-out fold.

Fixed definitions shared by every dataset (no per-dataset thresholds):
  * TP: same class and mask IoU >= 0.5, greedy by confidence (COCO AP50 rule).
  * Operating point: per-fold confidence that maximises F1 on the held-out fold.
  * FN type: misclassified | poorly_localized | missed.
  * FP type: duplicate | class_confusion (IoU >= 0.5 with a GT),
             partial (majority of FP pixels lie on GT), background (otherwise).
  * Only background FPs are exported as distractors, with GT pixels removed, and
    only if their area lies within the [min, max] GT area of the dataset.


"""

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================
DATASET = "armbench"  # "armbench" | "d2s"

PRESETS = {
    "armbench": {
        "image_dir": "**/train_images",
        "labelme_dir": "**/train_annotations",
        "classes": ["Object"],
        "out_root": "**/armbench",
        # Optional real validation split, used only for best.pt selection.
        "val_image_dir": "**/val_images",
        "val_labelme_dir": "**/val_annotations",
        # Optional regex on the file stem; images sharing group(1) stay in one fold.
        "group_regex": None,
    },
    "d2s": {
        "image_dir": "**/train/images",
        "labelme_dir": "**/train/jsons",
        "classes": "**/classes.json",
        "out_root": "**/d2s",
        "val_image_dir": "**/val/images",
        "val_labelme_dir": "**/val/jsons",
        # D2S_SSSSVV: SSSS = scene, VV = 30 views (rotations x lights) of the same scene.
        "group_regex": r"D2S_(\d{4})\d{2}",
    },
}

K_FOLDS = 5
SPLIT_SEED = 42

# Ultralytics downloads the pretrained weights automatically if not found locally.
BASE_MODEL = "yolo11s-seg.pt"

INFER_BATCH = 8  # lower if GPU memory runs out
DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = torch.cuda.is_available()

# Baseline YOLO11 recipe; epochs=100 only for fold (mining) models, the final training keeps its own recipe.
TRAIN_ARGS = {
    "epochs": 100, "imgsz": 1024, "batch": 8, "device": DEVICE, "workers": 4,
    "cache": False, "seed": 42, "deterministic": True, "pretrained": True, "amp": True,
    "save": True, "save_period": 10, "val": True, "plots": True, "patience": 100,
    "optimizer": "AdamW", "lr0": 0.002, "lrf": 0.01, "momentum": 0.9,
    "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0, "cos_lr": False,
    "mosaic": 1.0, "close_mosaic": 10, "mixup": 0.0, "copy_paste": 0.0,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
    "fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "erasing": 0.4, "auto_augment": "randaugment",
}

DISTRACTOR_MASK_LABEL = "Distractor"
DISTRACTOR_BOX_LABEL = "DistractorBox"

# Standard evaluation conventions (COCO / Ultralytics val), not tuning knobs.
MATCH_IOU = 0.5
MIN_CONF = 0.001
MAX_DET = 1000

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


# ============================================================
# DATA LOADING
# ============================================================
def load_classes(spec):
    if isinstance(spec, (list, tuple)):
        return [str(c) for c in spec]
    with open(spec, "r") as f:
        data = json.load(f)
    names = data["names"] if isinstance(data, dict) else data
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def find_image(image_dir, json_path, labelme_data):
    image_path_value = labelme_data.get("imagePath")
    if image_path_value:
        p = image_dir / Path(image_path_value).name
        if p.exists():
            return p
    for ext in IMAGE_EXTS:
        p = image_dir / f"{json_path.stem}{ext}"
        if p.exists():
            return p
    return None


def build_items(image_dir, labelme_dir, name_to_id):
    image_dir, labelme_dir = Path(image_dir), Path(labelme_dir)
    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {labelme_dir}")

    items = []
    ignored = Counter()
    for json_path in tqdm(json_files, desc="Loading annotations"):
        with open(json_path, "r") as f:
            data = json.load(f)
        data["imageData"] = None

        image_path = find_image(image_dir, json_path, data)
        if image_path is None:
            ignored["<missing image>"] += 1
            continue

        h, w = data.get("imageHeight"), data.get("imageWidth")
        if h is None or w is None:
            img = cv2.imread(str(image_path))
            if img is None:
                ignored["<unreadable image>"] += 1
                continue
            h, w = img.shape[:2]

        gts = []
        for shape_index, shape in enumerate(data.get("shapes", [])):
            label = str(shape.get("label"))
            points = shape.get("points", [])
            if label not in name_to_id:
                ignored[label] += 1
                continue
            if len(points) < 3:
                continue
            gts.append({
                "shape_index": shape_index,
                "shape": shape,
                "label": label,
                "cls": name_to_id[label],
                "points": points,
                "area": float(abs(cv2.contourArea(np.asarray(points, np.float32)))),
            })

        items.append({
            "stem": json_path.stem,
            "json": json_path,
            "image": image_path,
            "h": int(h),
            "w": int(w),
            "gts": gts,
            "labelme": data,
        })

    if ignored:
        print("Ignored shapes/files (not a model class or unusable):", dict(ignored))
    return items


def assign_folds(items, k, seed, group_regex):
    def group_key(stem):
        if group_regex:
            m = re.search(group_regex, stem)
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return stem

    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[group_key(item["stem"])].append(i)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda g: -len(groups[g]))  # stable sort keeps shuffle among equal sizes

    sizes = [0] * k
    fold_of = [None] * len(items)
    for g in keys:
        f = int(np.argmin(sizes))
        for i in groups[g]:
            fold_of[i] = f
        sizes[f] += len(groups[g])
    return fold_of


# ============================================================
# YOLO DATASETS + TRAINING
# ============================================================
def write_yolo_label(item, path):
    lines = []
    for gt in item["gts"]:
        pts = np.asarray(gt["points"], dtype=np.float64)
        pts[:, 0] = np.clip(pts[:, 0] / item["w"], 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / item["h"], 0.0, 1.0)
        lines.append(" ".join([str(gt["cls"])] + [f"{v:.6f}" for v in pts.reshape(-1)]))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def link_split(items, root, split):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    # Generated split folders; rebuilt so a changed fold assignment leaves no stale files.
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for item in items:
        (img_dir / item["image"].name).symlink_to(item["image"].resolve())
        write_yolo_label(item, lbl_dir / f"{item['image'].stem}.txt")


def write_data_yaml(path, root, val_path, names):
    cfg = {
        "path": str(root),
        "train": "images/train",
        "val": val_path,
        "names": {i: n for i, n in enumerate(names)},
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def fold_weights(run_root, k, use_last):
    return run_root / f"fold{k}" / "weights" / ("last.pt" if use_last else "best.pt")


def train_fold(k, data_yaml, base_model, train_args, run_root, use_last):
    run_dir = run_root / f"fold{k}"
    weights = fold_weights(run_root, k, use_last)
    done_marker = run_dir / "DONE"

    if done_marker.exists() and weights.exists():
        print(f"[fold {k}] trained model found, skipping training.")
        return weights

    last = run_dir / "weights" / "last.pt"
    if last.exists():
        print(f"[fold {k}] resuming interrupted training from {last}")
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as e:
            # Raised when the run already reached its final epoch.
            print(f"[fold {k}] nothing to resume: {e}")
    else:
        YOLO(base_model).train(
            data=str(data_yaml),
            project=str(run_root),
            name=f"fold{k}",
            exist_ok=True,
            **train_args,
        )
    if not weights.exists():
        raise FileNotFoundError(f"Expected weights not found: {weights}")
    done_marker.touch()
    return weights


# ============================================================
# GEOMETRY (original image <-> letterboxed inference frame)
# ============================================================
def letterbox_params(h, w, mh, mw):
    # Same formula as ultralytics.utils.ops.scale_boxes.
    gain = min(mh / h, mw / w)
    pad_x = round((mw - round(w * gain)) / 2 - 0.1)
    pad_y = round((mh - round(h * gain)) / 2 - 0.1)
    return gain, pad_x, pad_y


def polygon_to_infer_mask(points, gain, pad_x, pad_y, mh, mw):
    mask = np.zeros((mh, mw), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32) * gain
    pts += np.array([pad_x, pad_y], dtype=np.float32)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def box_to_infer_mask(box, gain, pad_x, pad_y, mh, mw):
    x1, y1, x2, y2 = box
    c1 = int(np.clip(np.floor(x1 * gain + pad_x), 0, mw))
    r1 = int(np.clip(np.floor(y1 * gain + pad_y), 0, mh))
    c2 = int(np.clip(np.ceil(x2 * gain + pad_x), 0, mw))
    r2 = int(np.clip(np.ceil(y2 * gain + pad_y), 0, mh))
    mask = np.zeros((mh, mw), dtype=bool)
    mask[r1:r2, c1:c2] = True
    return mask


def region_to_polygon(region, gain, pad_x, pad_y, h, w):
    """Largest connected region -> polygon in original image coordinates."""
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0.0

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return None, 0.0

    pts = cnt.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = np.clip((pts[:, 0] - pad_x) / gain, 0, w - 1)
    pts[:, 1] = np.clip((pts[:, 1] - pad_y) / gain, 0, h - 1)
    area = float(abs(cv2.contourArea(pts)))
    if area <= 0:
        return None, 0.0
    return pts.tolist(), area


# ============================================================
# PER-IMAGE ANALYSIS
# ============================================================
@torch.inference_mode()
def analyze(result, item):
    gts = item["gts"]
    G = len(gts)
    gt_cls = np.array([g["cls"] for g in gts], dtype=int)

    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return {
            "G": G, "N": 0, "gt_cls": gt_cls,
            "conf": np.zeros(0), "cls": np.zeros(0, dtype=int),
            "match": np.zeros(0, dtype=int), "gt_hit": np.zeros(G, dtype=bool),
        }

    masks = result.masks.data  # (N, mh, mw) uint8 in the letterboxed frame
    N, mh, mw = masks.shape
    conf = result.boxes.conf.float().cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.float().cpu().numpy()

    gain, pad_x, pad_y = letterbox_params(item["h"], item["w"], mh, mw)

    gt_masks = np.zeros((G, mh, mw), dtype=np.uint8)
    for g, gt in enumerate(gts):
        gt_masks[g] = polygon_to_infer_mask(gt["points"], gain, pad_x, pad_y, mh, mw)
    gt_union = gt_masks.any(axis=0)

    gt_t = torch.from_numpy(gt_masks.reshape(G, mh * mw)).to(masks.device, torch.float32)
    union_t = torch.from_numpy(gt_union.reshape(-1).astype(np.float32)).to(masks.device)
    gt_area = gt_t.sum(1)

    iou = np.zeros((G, N), dtype=np.float32)
    coverage = np.zeros(N, dtype=np.float32)
    for s in range(0, N, 256):
        p = masks[s:s + 256].reshape(-1, mh * mw).float()
        area = p.sum(1)
        coverage[s:s + len(p)] = ((p @ union_t) / area.clamp(min=1)).cpu().numpy()
        if G:
            inter = gt_t @ p.T
            union = gt_area[:, None] + area[None, :] - inter
            iou[:, s:s + len(p)] = (inter / union.clamp(min=1)).cpu().numpy()

    # Greedy, confidence-ordered, class-aware matching (COCO style).
    match = np.full(N, -1, dtype=int)
    gt_hit = np.zeros(G, dtype=bool)
    if G:
        for j in np.argsort(-conf, kind="stable"):
            cand = (gt_cls == cls[j]) & ~gt_hit & (iou[:, j] >= MATCH_IOU)
            if cand.any():
                g = int(np.argmax(np.where(cand, iou[:, j], -1.0)))
                gt_hit[g] = True
                match[j] = g

    return {
        "G": G, "N": N, "gt_cls": gt_cls,
        "conf": conf, "cls": cls, "boxes": boxes, "masks": masks,
        "iou": iou, "coverage": coverage, "gt_union": gt_union,
        "match": match, "gt_hit": gt_hit,
        "frame": (gain, pad_x, pad_y, mh, mw),
    }


def type_false_negative(g, a):
    if a["N"] == 0:
        return "missed"
    row = a["iou"][g]
    other_class = a["cls"] != a["gt_cls"][g]
    if (row[other_class] >= MATCH_IOU).any():
        return "misclassified"
    if (row[~other_class] > 0).any():
        return "poorly_localized"
    return "missed"


def type_false_positive(j, a):
    best_gt, best_iou = None, 0.0
    if a["G"]:
        best_gt = int(np.argmax(a["iou"][:, j]))
        best_iou = float(a["iou"][best_gt, j])
        if best_iou >= MATCH_IOU:
            same = a["gt_cls"][best_gt] == a["cls"][j]
            return ("duplicate" if same else "class_confusion"), best_gt, best_iou
    if a["coverage"][j] >= 0.5:
        return "partial", best_gt, best_iou
    return "background", best_gt, best_iou


def f1_optimal_conf(confs, tps, n_gt):
    if len(confs) == 0 or n_gt == 0:
        return MIN_CONF
    order = np.argsort(-confs, kind="stable")
    c = confs[order]
    tp_cum = np.cumsum(tps[order].astype(np.float64))
    precision = tp_cum / np.arange(1, len(c) + 1)
    recall = tp_cum / n_gt
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(c[int(np.argmax(f1))])


def predict_items(model, items, conf, imgsz):
    for s in range(0, len(items), INFER_BATCH):
        chunk = items[s:s + INFER_BATCH]
        results = model.predict(
            [str(it["image"]) for it in chunk],
            conf=conf,
            imgsz=imgsz,
            device=DEVICE,
            half=HALF,
            max_det=MAX_DET,
            verbose=False,
        )
        for item, result in zip(chunk, results):
            yield item, result


def labelme_shape(label, points, group_id=None):
    return {
        "label": label,
        "points": points,
        "group_id": group_id,
        "description": "",
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def labelme_file(item, shapes):
    data = dict(item["labelme"])
    data.update({
        "shapes": shapes,
        "imagePath": item["image"].name,
        "imageHeight": item["h"],
        "imageWidth": item["w"],
        "imageData": None,
    })
    return data


# ============================================================
# MINING ONE FOLD
# ============================================================
def mine_fold(k, weights, held_items, imgsz, area_bounds, out, names, rows):
    model = YOLO(str(weights))

    # Pass 1: full precision/recall curve on the held-out fold -> F1-optimal confidence.
    confs, tps, n_gt = [], [], 0
    for item, result in tqdm(predict_items(model, held_items, MIN_CONF, imgsz),
                             total=len(held_items), desc=f"[fold {k}] calibrate"):
        a = analyze(result, item)
        confs.append(a["conf"])
        tps.append(a["match"] >= 0)
        n_gt += a["G"]
        del a, result
    tau = f1_optimal_conf(np.concatenate(confs), np.concatenate(tps), n_gt)

    # Ultralytics keeps conf > threshold, so step just below tau to include it.
    thr = float(np.nextafter(np.float32(tau), np.float32(0)))

    stats = Counter()
    fp_types = Counter()
    fn_types = Counter()
    area_lo, area_hi = area_bounds

    # Pass 2: predictions at the operating point -> typed failures.
    for item, result in tqdm(predict_items(model, held_items, thr, imgsz),
                             total=len(held_items), desc=f"[fold {k}] mine"):
        a = analyze(result, item)
        stats["images"] += 1
        stats["gt"] += a["G"]
        stats["tp"] += int((a["match"] >= 0).sum())

        fn_shapes, fn_meta = [], []
        for g, gt in enumerate(item["gts"]):
            if a["gt_hit"][g]:
                continue
            fn_type = type_false_negative(g, a)
            fn_types[fn_type] += 1
            fn_shapes.append(gt["shape"])
            fn_meta.append({
                "shape_index": gt["shape_index"],
                "label": gt["label"],
                "type": fn_type,
                "area": gt["area"],
            })
            rows.append({
                "fold": k, "tau": tau, "image": item["image"].name, "kind": "FN",
                "type": fn_type, "gt_label": gt["label"], "pred_label": None,
                "conf": None, "best_iou_any_gt": None, "gt_coverage": None,
                "area_px": gt["area"], "exported": True,
            })

        fp_shapes, fp_meta = [], []
        for j in np.where(a["match"] < 0)[0]:
            fp_type, best_gt, best_iou = type_false_positive(j, a)
            fp_types[fp_type] += 1
            exported, area = False, None

            if fp_type == "background":
                gain, pad_x, pad_y, mh, mw = a["frame"]
                # Remove any GT pixels so no real object is pasted without a label.
                mask_region = (a["masks"][j] > 0).cpu().numpy() & ~a["gt_union"]
                pts, area = region_to_polygon(mask_region, gain, pad_x, pad_y, item["h"], item["w"])
                if pts is not None and area_lo <= area <= area_hi:
                    box_region = box_to_infer_mask(a["boxes"][j], gain, pad_x, pad_y, mh, mw) & ~a["gt_union"]
                    box_pts, _ = region_to_polygon(box_region, gain, pad_x, pad_y, item["h"], item["w"])
                    gid = len(fp_meta)
                    fp_shapes.append(labelme_shape(DISTRACTOR_MASK_LABEL, pts, gid))
                    if box_pts is not None:
                        fp_shapes.append(labelme_shape(DISTRACTOR_BOX_LABEL, box_pts, gid))
                    exported = True

            pred_label = names[int(a["cls"][j])]
            meta = {
                "type": fp_type,
                "pred_label": pred_label,
                "conf": float(a["conf"][j]),
                "best_iou_any_gt": best_iou,
                "best_gt_label": None if best_gt is None else item["gts"][best_gt]["label"],
                "gt_coverage": float(a["coverage"][j]),
                "area_px": area,
                "exported": exported,
            }
            if exported:
                fp_meta.append(meta)
            rows.append({
                "fold": k, "tau": tau, "image": item["image"].name, "kind": "FP",
                "type": fp_type, "gt_label": meta["best_gt_label"], "pred_label": pred_label,
                "conf": meta["conf"], "best_iou_any_gt": best_iou,
                "gt_coverage": meta["gt_coverage"], "area_px": area, "exported": exported,
            })

        stats["fn"] += len(fn_shapes)
        stats["fp_exported"] += len(fp_meta)

        with open(out["metadata"] / f"{item['stem']}.json", "w") as f:
            json.dump({
                "image": item["image"].name, "fold": k, "tau": tau,
                "num_gt": a["G"], "num_pred": a["N"],
                "false_negatives": fn_meta,
                "exported_distractors": fp_meta,
            }, f, indent=2)

        if fn_shapes or fp_shapes:
            shutil.copy2(item["image"], out["images"] / item["image"].name)
        if fn_shapes:
            with open(out["json_fn"] / item["json"].name, "w") as f:
                json.dump(labelme_file(item, fn_shapes), f, indent=2)
        if fp_shapes:
            with open(out["json_fp"] / item["json"].name, "w") as f:
                json.dump(labelme_file(item, fp_shapes), f, indent=2)

        del a, result

    fold_summary = {
        "fold": k,
        "tau_f1": tau,
        "images": stats["images"],
        "gt": stats["gt"],
        "tp": stats["tp"],
        "fn": stats["fn"],
        "fn_by_type": dict(fn_types),
        "fp_by_type": dict(fp_types),
        "fp_exported": stats["fp_exported"],
    }
    print(json.dumps(fold_summary, indent=2))
    return fold_summary


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Out-of-fold failure mining")
    p.add_argument("--dataset", default=DATASET, choices=sorted(PRESETS))
    p.add_argument("--stage", default="all", choices=["all", "prepare", "train", "mine"],
                   help="prepare: fold datasets; train: fold models; mine: export failures")
    p.add_argument("--fold", type=int, default=None,
                   help="train only this fold (one SLURM job per fold)")
    p.add_argument("--base-model", default=None,
                   help="architecture override, e.g. yolov8s-seg.pt (default: BASE_MODEL)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PRESETS[args.dataset]

    names = load_classes(cfg["classes"])
    name_to_id = {n: i for i, n in enumerate(names)}

    base_model = args.base_model or BASE_MODEL
    train_args = dict(TRAIN_ARGS)
    imgsz = train_args.get("imgsz", 1024)

    # One output folder per mining architecture, so several can coexist.
    out_root = Path(cfg["out_root"]) / Path(base_model).stem
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.dataset} | classes: {len(names)} | K={K_FOLDS} | "
          f"model: {base_model} | stage: {args.stage}")
    print("Output:", out_root)

    items = build_items(cfg["image_dir"], cfg["labelme_dir"], name_to_id)
    folds = assign_folds(items, K_FOLDS, SPLIT_SEED, cfg["group_regex"])
    print("Images:", len(items), "| per fold:", dict(sorted(Counter(folds).items())))

    # Without a real val split the held-out fold is the val set, so model selection
    # must not look at it: use last.pt and disable early stopping.
    external_val = bool(cfg["val_image_dir"] and cfg["val_labelme_dir"])
    if not external_val:
        train_args["patience"] = 0

    run_root = out_root / "fold_runs"
    fold_ids = range(K_FOLDS) if args.fold is None else [args.fold]

    # ---------------- prepare ----------------
    if args.stage in ("all", "prepare"):
        with open(out_root / "folds.json", "w") as f:
            json.dump({it["image"].name: int(fd) for it, fd in zip(items, folds)}, f, indent=2)

        if external_val:
            val_items = build_items(cfg["val_image_dir"], cfg["val_labelme_dir"], name_to_id)
            link_split(val_items, out_root / "yolo_val", "val")
            val_path = str(out_root / "yolo_val" / "images" / "val")
        else:
            val_path = "images/val"

        for k in range(K_FOLDS):
            fold_root = out_root / "fold_data" / f"fold{k}"
            link_split([it for it, fd in zip(items, folds) if fd != k], fold_root, "train")
            if not external_val:
                link_split([it for it, fd in zip(items, folds) if fd == k], fold_root, "val")
            write_data_yaml(fold_root / "data.yaml", fold_root, val_path, names)
        print("Fold datasets ready.")

    # ---------------- train ----------------
    if args.stage in ("all", "train"):
        for k in fold_ids:
            data_yaml = out_root / "fold_data" / f"fold{k}" / "data.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(f"{data_yaml} missing; run --stage prepare first.")
            train_fold(k, data_yaml, base_model, train_args, run_root, use_last=not external_val)

    if args.stage not in ("all", "mine"):
        return

    # ---------------- mine ----------------
    weights_per_fold = {}
    for k in range(K_FOLDS):
        w = fold_weights(run_root, k, use_last=not external_val)
        if not ((run_root / f"fold{k}" / "DONE").exists() and w.exists()):
            raise FileNotFoundError(f"Fold {k} is not trained yet ({w}).")
        weights_per_fold[k] = w

    gt_areas = [gt["area"] for it in items for gt in it["gts"] if gt["area"] > 0]
    area_bounds = (float(min(gt_areas)), float(max(gt_areas)))
    print("Distractor area bounds from GT (px):", area_bounds)

    mined = out_root / "mined"
    # Fully regenerated output, so it is cleared to avoid mixing with older runs.
    if mined.exists():
        shutil.rmtree(mined)
    out = {
        "images": mined / "images",
        "json_fn": mined / "json_fn_only",
        "json_fp": mined / "json_fp_only_distractor",
        "metadata": mined / "metadata",
    }
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)

    rows, fold_summaries = [], []
    for k in range(K_FOLDS):
        held_items = [it for it, fd in zip(items, folds) if fd == k]
        fold_summaries.append(
            mine_fold(k, weights_per_fold[k], held_items, imgsz, area_bounds, out, names, rows)
        )

    df = pd.DataFrame(rows)
    df.to_csv(mined / "oof_failures.csv", index=False)

    n_images = sum(s["images"] for s in fold_summaries)
    n_gt = sum(s["gt"] for s in fold_summaries)
    n_tp = sum(s["tp"] for s in fold_summaries)
    n_fn = sum(s["fn"] for s in fold_summaries)
    fp_by_type = Counter()
    fn_by_type = Counter()
    for s in fold_summaries:
        fp_by_type.update(s["fp_by_type"])
        fn_by_type.update(s["fn_by_type"])
    n_fp = sum(fp_by_type.values())
    n_bg = fp_by_type.get("background", 0)
    n_bg_exported = sum(s["fp_exported"] for s in fold_summaries)

    fp_df = df[df["kind"] == "FP"] if len(df) else df
    fn_df = df[df["kind"] == "FN"] if len(df) else df

    stats = {
        "dataset": args.dataset,
        "mining_model": base_model,
        "k_folds": K_FOLDS,
        "definitions": {
            "match": f"same class and mask IoU >= {MATCH_IOU}, greedy by confidence",
            "operating_point": "per-fold F1-optimal confidence on the held-out fold",
            "fp_types": "duplicate/class_confusion: IoU>=0.5 with a GT; partial: >=50% of FP pixels on GT; background: otherwise",
            "distractor_area_bounds_px": list(area_bounds),
        },
        "images": n_images,
        "gt": n_gt,
        "tp": n_tp,
        "fn": n_fn,
        "fp": n_fp,
        "fn_by_type": dict(fn_by_type),
        "fp_by_type": dict(fp_by_type),
        "background_fp_exported": n_bg_exported,
        "recall": n_tp / max(n_gt, 1),
        "precision": n_tp / max(n_tp + n_fp, 1),
        # Rates for adaptive paste counts (see run.py integration).
        "q_fn": n_fn / max(n_gt, 1),
        "q_fp_background": n_bg / max(n_tp + n_fp, 1),
        "background_fp_per_image": n_bg / max(n_images, 1),
        "background_fp_per_gt": n_bg / max(n_gt, 1),
        "fn_by_class": fn_df["gt_label"].value_counts().to_dict() if len(fn_df) else {},
        "fp_by_pred_class": fp_df["pred_label"].value_counts().to_dict() if len(fp_df) else {},
        "folds": fold_summaries,
    }
    with open(mined / "oof_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n====================================")
    print("OOF mining finished")
    print("====================================")
    for key in ["images", "gt", "tp", "fn", "fp", "fn_by_type", "fp_by_type",
                "background_fp_exported", "recall", "precision",
                "q_fn", "q_fp_background", "background_fp_per_image"]:
        print(f"{key:>24}: {stats[key]}")
    print("Output:", mined)


if __name__ == "__main__":
    main()
