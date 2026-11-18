from pathlib import Path
import gc

import pandas as pd
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DATA = Path(
    "path/to/dataset.yaml"
)

BASE_MODEL = "yolov8s-seg.pt"
# You can also select the following models:
# BASE_MODEL = "yolo11s-seg.pt"
# BASE_MODEL = "yolo26s-seg.pt"



# Models will be saved under:
# runs/segment/runs/instance_segmentation_seedXX/weights/best.pt
PROJECT = Path(
    "path/to/runs/segment/runs"
)

SEEDS = [42, 43, 44]

IMG_SIZE = 1024
TRAIN_BATCH = 8
TEST_BATCH = 4
DEVICE = 0
WORKERS = 4
TEST_SPLIT = "test"


def train_seed(seed):
    run_name = f"instance_segmentation_seed{seed}"

    best_pt = (
        PROJECT
        / run_name
        / "weights"
        / "best.pt"
    )

    # Skip training if this seed was already completed
    if best_pt.exists():
        print("\n" + "=" * 70)
        print(f"Seed {seed} has already been trained.")
        print(f"Using: {best_pt}")
        print("=" * 70)
        return best_pt

    print("\n" + "=" * 70)
    print(f"TRAINING SEED {seed}")
    print("=" * 70)

    # Fresh pretrained model for every seed
    model = YOLO(BASE_MODEL)

    model.train(
        data=str(DATA),
        epochs=200,
        imgsz=IMG_SIZE,
        batch=TRAIN_BATCH,
        device=DEVICE,
        workers=WORKERS,

        project=str(PROJECT),
        name=run_name,
        exist_ok=False,

        cache=False,
        seed=seed,
        deterministic=True,
        pretrained=True,
        amp=True,

        save=True,
        save_period=10,
        val=True,
        plots=True,
        patience=200,

        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        cos_lr=False,

        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        auto_augment="randaugment",

        single_cls=False,
        overlap_mask=True,
        mask_ratio=4,
    )

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not best_pt.exists():
        raise FileNotFoundError(
            f"Training finished, but best.pt was not found:\n{best_pt}"
        )

    return best_pt


def evaluate_seed(seed, best_pt):
    print("\n" + "-" * 70)
    print(f"EVALUATING SEED {seed} ON TEST SET")
    print(f"Checkpoint: {best_pt}")
    print("-" * 70)

    model = YOLO(str(best_pt))

    metrics = model.val(
        data=str(DATA),
        split=TEST_SPLIT,
        imgsz=IMG_SIZE,
        batch=TEST_BATCH,
        device=DEVICE,
        workers=WORKERS,

        project=str(PROJECT / "test_results"),
        name=f"test_seed{seed}",
        exist_ok=True,
    )

    result = {
        "seed": seed,

        "box_P": float(metrics.box.mp),
        "box_R": float(metrics.box.mr),
        "box_mAP50": float(metrics.box.map50),
        "box_mAP50_95": float(metrics.box.map),

        "mask_P": float(metrics.seg.mp),
        "mask_R": float(metrics.seg.mr),
        "mask_mAP50": float(metrics.seg.map50),
        "mask_mAP50_95": float(metrics.seg.map),
    }

    del model
    del metrics
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    PROJECT.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in SEEDS:
        best_pt = train_seed(seed)
        result = evaluate_seed(seed, best_pt)
        rows.append(result)

        # Save completed results after each seed
        pd.DataFrame(rows).to_csv(
            PROJECT / "test_results_partial.csv",
            index=False,
        )

    df = pd.DataFrame(rows)

    if len(df) != len(SEEDS):
        raise RuntimeError(
            f"Expected {len(SEEDS)} seed results, "
            f"but obtained {len(df)}."
        )

    metric_df = df.drop(columns=["seed"])

    summary_table = pd.DataFrame({
        "mean": metric_df.mean(),
        # Sample standard deviation
        "std": metric_df.std(ddof=1),
    })

    summary_table["mean ± std"] = summary_table.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}",
        axis=1,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 70)
    print("PER-SEED TEST RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("MEAN ± STANDARD DEVIATION")
    print("=" * 70)
    print(summary_table[["mean ± std"]].to_string())

    per_seed_file = PROJECT / "test_results_per_seed.csv"
    summary_file = PROJECT / "test_results_mean_std.csv"

    df.to_csv(per_seed_file, index=False)
    summary_table.to_csv(summary_file)

    print("\nSaved:")
    print(per_seed_file)
    print(summary_file)


if __name__ == "__main__":
    main()