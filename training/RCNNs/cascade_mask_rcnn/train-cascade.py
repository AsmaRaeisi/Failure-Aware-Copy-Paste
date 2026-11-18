from __future__ import annotations

import csv
import gc
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


# ============================================================
# MMDETECTION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

MMDET_ROOT = Path(
    "**/mmdetection"
)


# ============================================================
# DATASET PATHS
# ============================================================
#
# Current real-data baseline:
#
# /**/real/
# ├── images/
# ├── train.json
# ├── val.json
# └── test.json
#
# For later experiments, only change:
#   TRAIN_IMAGE_DIR
#   TRAIN_ANN_FILE
#   EXPERIMENT_NAME
#
# Validation and test remain fixed to the real dataset.
# ============================================================


# ------------------------------------------------------------
# TRAINING DATA
# ------------------------------------------------------------

TRAIN_IMAGE_DIR = Path(
    "**/images"
)

TRAIN_ANN_FILE = Path(
    "**/train.json"
)


# ------------------------------------------------------------
# VALIDATION DATA
# ------------------------------------------------------------

VAL_IMAGE_DIR = Path(
    "**/val/images"
)

VAL_ANN_FILE = Path(
    "**/val.json"
)


# ------------------------------------------------------------
# TEST DATA
# ------------------------------------------------------------

TEST_IMAGE_DIR = Path(
    "**/test/images"
)

TEST_ANN_FILE = Path(
    "**/test.json"
)


# ============================================================
# MODEL
# ============================================================

# Active architecture: Mask R-CNN with ResNet-50 and FPN.
BASE_CONFIG = (
    MMDET_ROOT
    / "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"
)

PRETRAINED_CHECKPOINT = (
    "https://download.openmmlab.com/mmdetection/v2.0/"
    "cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/"
    "cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
)



# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

EXPERIMENT_NAME = "cascade_maskrcnn_r50_fpn_real_1024_24e"

RESULT_ROOT = (
    SCRIPT_DIR
    / "work_dirs"
    / EXPERIMENT_NAME
)

SEEDS = (42, 43, 44)

IMAGE_SCALE = (1024, 1024)

MAX_EPOCHS = 24

TRAIN_BATCH_SIZE = 4

NUM_WORKERS = 4

# Standard Mask R-CNN LR is 0.02 for total batch size 16.
# For one GPU with batch size 4:
# 0.02 * 4 / 16 = 0.005
LEARNING_RATE = 0.005

# Standard 24-epoch (2x-style) learning-rate milestones.
LR_MILESTONES = [18, 22]


# ============================================================
# DATASET VALIDATION
# ============================================================

def load_coco_json(
    annotation_file: Path,
) -> dict[str, Any]:
    with annotation_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def read_coco_classes(
    annotation_file: Path,
) -> tuple[str, ...]:
    coco = load_coco_json(
        annotation_file
    )

    categories = sorted(
        coco["categories"],
        key=lambda category: category["id"],
    )

    return tuple(
        category["name"]
        for category in categories
    )


def inspect_coco_split(
    split_name: str,
    image_dir: Path,
    annotation_file: Path,
) -> tuple[str, ...]:
    coco = load_coco_json(
        annotation_file
    )

    classes = read_coco_classes(
        annotation_file
    )

    image_sizes = sorted(
        {
            (
                image["width"],
                image["height"],
            )
            for image in coco["images"]
        }
    )

    category_ids = sorted(
        {
            annotation["category_id"]
            for annotation in coco["annotations"]
        }
    )

    missing_image_files = []

    for image in coco["images"]:
        image_path = image_dir / image["file_name"]

        if not image_path.exists():
            missing_image_files.append(
                str(image_path)
            )

            if len(missing_image_files) >= 10:
                break

    if missing_image_files:
        raise FileNotFoundError(
            f"{split_name} contains image paths that do not exist. "
            "First missing paths:\n"
            + "\n".join(missing_image_files)
        )

    print(
        f"{split_name}: "
        f"{len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations"
    )

    print(
        f"{split_name} image directory: "
        f"{image_dir}"
    )

    print(
        f"{split_name} annotation file: "
        f"{annotation_file}"
    )

    print(
        f"{split_name} classes: "
        f"{classes}"
    )

    print(
        f"{split_name} category IDs used: "
        f"{category_ids}"
    )

    print(
        f"{split_name} image sizes: "
        f"{image_sizes[:10]}"
    )

    return classes


def validate_dataset_paths() -> tuple[str, ...]:
    required_paths = [
        MMDET_ROOT,
        BASE_CONFIG,
        TRAIN_IMAGE_DIR,
        TRAIN_ANN_FILE,
        VAL_IMAGE_DIR,
        VAL_ANN_FILE,
        TEST_IMAGE_DIR,
        TEST_ANN_FILE,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "The following required paths do not exist:\n"
            + "\n".join(
                str(path)
                for path in missing_paths
            )
        )

    train_classes = inspect_coco_split(
        split_name="TRAIN",
        image_dir=TRAIN_IMAGE_DIR,
        annotation_file=TRAIN_ANN_FILE,
    )

    val_classes = inspect_coco_split(
        split_name="VAL",
        image_dir=VAL_IMAGE_DIR,
        annotation_file=VAL_ANN_FILE,
    )

    test_classes = inspect_coco_split(
        split_name="TEST",
        image_dir=TEST_IMAGE_DIR,
        annotation_file=TEST_ANN_FILE,
    )

    if not (
        train_classes
        == val_classes
        == test_classes
    ):
        raise ValueError(
            "Class definitions differ between the splits:\n"
            f"train={train_classes}\n"
            f"val={val_classes}\n"
            f"test={test_classes}"
        )

    return train_classes


# ============================================================
# MODEL CONFIGURATION HELPERS
# ============================================================

def set_num_classes(
    node: Any,
    num_classes: int,
) -> None:
    """
    Recursively update every num_classes entry.

    This supports standard Mask R-CNN and multi-stage models
    such as Cascade Mask R-CNN.
    """

    if isinstance(node, dict):
        for key, value in node.items():
            if key == "num_classes":
                node[key] = num_classes
            else:
                set_num_classes(
                    value,
                    num_classes,
                )

    elif isinstance(node, (list, tuple)):
        for value in node:
            set_num_classes(
                value,
                num_classes,
            )


def configure_dataset(
    dataset_cfg: Any,
    image_dir: Path,
    annotation_file: Path,
    classes: tuple[str, ...],
    pipeline: list[dict[str, Any]],
    test_mode: bool,
) -> None:
    """
    Configure a COCO dataset using independent absolute paths
    for its image directory and annotation file.
    """

    dataset_cfg.data_root = ""

    dataset_cfg.ann_file = str(
        annotation_file.resolve()
    )

    dataset_cfg.data_prefix = {
        "img": str(
            image_dir.resolve()
        ) + "/"
    }

    dataset_cfg.metainfo = {
        "classes": classes
    }

    dataset_cfg.pipeline = pipeline

    dataset_cfg.test_mode = test_mode


# ============================================================
# BUILD CONFIG FOR ONE SEED
# ============================================================

def build_config(
    seed: int,
    classes: tuple[str, ...],
    seed_work_dir: Path,
) -> Config:
    cfg = Config.fromfile(
        str(BASE_CONFIG)
    )

    cfg.work_dir = str(
        seed_work_dir
        )

    # ========================================================
    # SAVE TRAINING AND VALIDATION METRICS
    # ========================================================

    cfg.visualizer = {
        "type": "DetLocalVisualizer",
        "vis_backends": [
            {
                "type": "LocalVisBackend",
            },
            {
                "type": "TensorboardVisBackend",
            },
        ],
        "name": f"visualizer_seed{seed}",
    }

    cfg.log_processor = {
        "type": "LogProcessor",
        "window_size": 50,
        "by_epoch": True,
    }


    cfg.load_from = (
        PRETRAINED_CHECKPOINT
    )

    cfg.resume = False

    cfg.randomness = {
        "seed": seed,
        "deterministic": False,
    }

    cfg.env_cfg.cudnn_benchmark = False

    set_num_classes(
        cfg.model,
        len(classes),
    )


    # ========================================================
    # TRAINING PIPELINE
    # ========================================================

    train_pipeline = [
        {
            "type": "LoadImageFromFile",
            "backend_args": None,
        },
        {
            "type": "LoadAnnotations",
            "with_bbox": True,
            "with_mask": True,
        },
        {
            "type": "Resize",
            "scale": IMAGE_SCALE,
            "keep_ratio": True,
        },
        {
            "type": "RandomFlip",
            "prob": 0.5,
        },
        {
            "type": "PackDetInputs",
        },
    ]


    # ========================================================
    # VALIDATION AND TEST PIPELINE
    # No random augmentation.
    # ========================================================

    evaluation_pipeline = [
        {
            "type": "LoadImageFromFile",
            "backend_args": None,
        },
        {
            "type": "Resize",
            "scale": IMAGE_SCALE,
            "keep_ratio": True,
        },
        {
            "type": "LoadAnnotations",
            "with_bbox": True,
            "with_mask": True,
        },
        {
            "type": "PackDetInputs",
            "meta_keys": (
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
            ),
        },
    ]

    cfg.train_pipeline = train_pipeline
    cfg.test_pipeline = evaluation_pipeline


    # ========================================================
    # DATASETS
    # ========================================================

    configure_dataset(
        dataset_cfg=cfg.train_dataloader.dataset,
        image_dir=TRAIN_IMAGE_DIR,
        annotation_file=TRAIN_ANN_FILE,
        classes=classes,
        pipeline=train_pipeline,
        test_mode=False,
    )

    configure_dataset(
        dataset_cfg=cfg.val_dataloader.dataset,
        image_dir=VAL_IMAGE_DIR,
        annotation_file=VAL_ANN_FILE,
        classes=classes,
        pipeline=evaluation_pipeline,
        test_mode=True,
    )

    configure_dataset(
        dataset_cfg=cfg.test_dataloader.dataset,
        image_dir=TEST_IMAGE_DIR,
        annotation_file=TEST_ANN_FILE,
        classes=classes,
        pipeline=evaluation_pipeline,
        test_mode=True,
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    cfg.train_dataloader.batch_size = (
        TRAIN_BATCH_SIZE
    )

    cfg.train_dataloader.num_workers = (
        NUM_WORKERS
    )

    cfg.train_dataloader.persistent_workers = True
    cfg.train_dataloader.pin_memory = True

    for dataloader in (
        cfg.val_dataloader,
        cfg.test_dataloader,
    ):
        dataloader.batch_size = 1
        dataloader.num_workers = NUM_WORKERS
        dataloader.persistent_workers = True
        dataloader.pin_memory = True


    # ========================================================
    # VALIDATION AND TEST EVALUATORS
    # ========================================================

    cfg.val_evaluator = {
        "type": "CocoMetric",
        "ann_file": str(
            VAL_ANN_FILE.resolve()
        ),
        "metric": [
            "bbox",
            "segm",
        ],
        "format_only": False,
        "classwise": False,
        "backend_args": None,
    }

    cfg.test_evaluator = {
        "type": "CocoMetric",
        "ann_file": str(
            TEST_ANN_FILE.resolve()
        ),
        "metric": [
            "bbox",
            "segm",
        ],
        "format_only": False,
        "classwise": False,
        "backend_args": None,
    }


    # ========================================================
    # TRAINING SCHEDULE
    # ========================================================

    cfg.train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": MAX_EPOCHS,
        "val_interval": 1,
    }

    cfg.val_cfg = {
        "type": "ValLoop"
    }

    cfg.test_cfg = {
        "type": "TestLoop"
    }


    # ========================================================
    # OPTIMIZER AND LEARNING-RATE SCHEDULE
    # ========================================================

    cfg.optim_wrapper.optimizer.lr = (
        LEARNING_RATE
    )

    cfg.auto_scale_lr = {
        "enable": False,
        "base_batch_size": 16,
    }

    cfg.param_scheduler = [
        {
            "type": "LinearLR",
            "start_factor": 0.001,
            "by_epoch": False,
            "begin": 0,
            "end": 500,
        },
        {
            "type": "MultiStepLR",
            "by_epoch": True,
            "begin": 0,
            "end": MAX_EPOCHS,
            "milestones": LR_MILESTONES,
            "gamma": 0.1,
        },
    ]


    # ========================================================
    # BEST VALIDATION CHECKPOINT
    # ========================================================

    cfg.default_hooks.logger.interval = 50

    cfg.default_hooks.checkpoint.interval = 1

    cfg.default_hooks.checkpoint.save_best = (
        "coco/segm_mAP"
    )

    cfg.default_hooks.checkpoint.rule = (
        "greater"
    )

    cfg.default_hooks.checkpoint.max_keep_ckpts = 1

    cfg.default_hooks.checkpoint.save_last = True

    return cfg


# ============================================================
# FIND BEST CHECKPOINT
# ============================================================

def find_best_checkpoint(
    runner: Runner,
    seed_work_dir: Path,
) -> Path:
    saved_checkpoint = (
        runner.message_hub.get_info(
            "best_ckpt"
        )
    )

    if saved_checkpoint:
        checkpoint_path = Path(
            str(saved_checkpoint)
        )

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                seed_work_dir
                / checkpoint_path
            )

        if checkpoint_path.exists():
            return checkpoint_path

    candidates = sorted(
        seed_work_dir.glob(
            "best_*.pth"
        ),
        key=lambda path: path.stat().st_mtime,
    )

    if not candidates:
        raise RuntimeError(
            "No best validation checkpoint was found in:\n"
            f"{seed_work_dir}"
        )

    return candidates[-1]


def clean_metric_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    if hasattr(value, "item"):
        return value.item()

    return str(value)


# ============================================================
# TRAIN AND TEST ONE SEED
# ============================================================

def train_and_test_seed(
    seed: int,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    seed_work_dir = (
        RESULT_ROOT
        / f"seed{seed}"
    )

    result_file = (
        seed_work_dir
        / "test_metrics.json"
    )


    # Skip a seed that was already completed successfully.
    if result_file.exists():
        print("\n" + "=" * 80)

        print(
            f"Seed {seed} was already completed."
        )

        print(
            f"Loading results from: "
            f"{result_file}"
        )

        with result_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)


    # Restart an incomplete seed from the beginning.
    if seed_work_dir.exists():
        shutil.rmtree(
            seed_work_dir
        )

    seed_work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("\n" + "=" * 80)

    print(
        f"TRAINING SEED {seed}"
    )

    print(
        f"Seed output directory: "
        f"{seed_work_dir}"
    )

    print("=" * 80)


    cfg = build_config(
        seed=seed,
        classes=classes,
        seed_work_dir=seed_work_dir,
    )

    runner = Runner.from_cfg(
        cfg
    )


    # Train on TRAIN and validate after every epoch.
    runner.train()


    # Find the best validation checkpoint.
    best_checkpoint = find_best_checkpoint(
        runner=runner,
        seed_work_dir=seed_work_dir,
    )

    print(
        f"\nBest validation checkpoint "
        f"for seed {seed}:"
    )

    print(
        best_checkpoint
    )


    # Load the best validation checkpoint.
    runner.load_checkpoint(
        str(best_checkpoint)
    )


    # Evaluate that checkpoint once on TEST.
    print(
        f"\nEvaluating seed {seed} "
        "on the test set..."
    )

    raw_test_metrics = (
        runner.test()
    )

    test_metrics = {
        metric_name:
            clean_metric_value(metric_value)
        for metric_name, metric_value
        in raw_test_metrics.items()
    }


    result = {
        "seed": seed,

        "train_image_dir": str(
            TRAIN_IMAGE_DIR
        ),

        "train_annotation_file": str(
            TRAIN_ANN_FILE
        ),

        "validation_image_dir": str(
            VAL_IMAGE_DIR
        ),

        "validation_annotation_file": str(
            VAL_ANN_FILE
        ),

        "test_image_dir": str(
            TEST_IMAGE_DIR
        ),

        "test_annotation_file": str(
            TEST_ANN_FILE
        ),

        "best_checkpoint": str(
            best_checkpoint
        ),

        "test_metrics": test_metrics,
    }


    with result_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )


    del runner

    gc.collect()

    torch.cuda.empty_cache()

    return result


# ============================================================
# METRIC HELPERS
# ============================================================

def numeric_metrics(
    result: dict[str, Any],
) -> dict[str, float]:
    return {
        metric_name: float(metric_value)
        for metric_name, metric_value
        in result["test_metrics"].items()
        if isinstance(
            metric_value,
            (int, float),
        )
        and not isinstance(
            metric_value,
            bool,
        )
    }


# ============================================================
# SAVE THREE-SEED RESULTS
# ============================================================

def write_final_summary(
    results: list[dict[str, Any]],
) -> None:
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(results) < 2:
        raise RuntimeError(
            "At least two completed seeds are required "
            "to calculate sample standard deviation."
        )

    numeric_results = {
        result["seed"]:
            numeric_metrics(result)
        for result in results
    }

    if not numeric_results:
        raise RuntimeError(
            "No numeric test metrics were returned."
        )

    metric_sets = [
        set(seed_metrics.keys())
        for seed_metrics
        in numeric_results.values()
    ]

    metric_keys = sorted(
        set.intersection(
            *metric_sets
        )
    )

    if not metric_keys:
        raise RuntimeError(
            "No common numeric metrics were found "
            "across the seeds."
        )


    # ========================================================
    # SAVE INDIVIDUAL SEED RESULTS
    # ========================================================

    individual_results_csv = (
        RESULT_ROOT
        / "test_metrics_all_seeds.csv"
    )

    with individual_results_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "seed",
            "best_checkpoint",
            *metric_keys,
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            seed = result["seed"]

            writer.writerow(
                {
                    "seed": seed,
                    "best_checkpoint":
                        result["best_checkpoint"],
                    **numeric_results[seed],
                }
            )


    # ========================================================
    # MEAN AND SAMPLE STANDARD DEVIATION
    # ========================================================

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for metric_name in metric_keys:
        values = [
            numeric_results[
                result["seed"]
            ][metric_name]
            for result in results
        ]

        summary[metric_name] = {
            "mean":
                statistics.fmean(values),

            # Sample standard deviation, denominator N - 1.
            "std":
                statistics.stdev(values),
        }


    # ========================================================
    # SAVE MEAN AND STD
    # ========================================================

    summary_json = (
        RESULT_ROOT
        / "test_metrics_mean_std.json"
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )


    summary_csv = (
        RESULT_ROOT
        / "test_metrics_mean_std.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "mean",
                "std",
            ],
        )

        writer.writeheader()

        for metric_name, values in summary.items():
            writer.writerow(
                {
                    "metric": metric_name,
                    "mean": values["mean"],
                    "std": values["std"],
                }
            )


    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    preferred_metric_order = [
        "coco/bbox_mAP",
        "coco/bbox_mAP_50",
        "coco/bbox_mAP_75",
        "coco/bbox_mAP_s",
        "coco/bbox_mAP_m",
        "coco/bbox_mAP_l",
        "coco/segm_mAP",
        "coco/segm_mAP_50",
        "coco/segm_mAP_75",
        "coco/segm_mAP_s",
        "coco/segm_mAP_m",
        "coco/segm_mAP_l",
    ]


    print("\n" + "=" * 80)

    print(
        "INDIVIDUAL TEST RESULTS"
    )

    print("=" * 80)

    for result in results:
        seed = result["seed"]

        print(
            f"\nSeed {seed}"
        )

        for metric_name in preferred_metric_order:
            if metric_name in numeric_results[seed]:
                print(
                    f"{metric_name:25s}: "
                    f"{numeric_results[seed][metric_name]:.4f}"
                )


    print("\n" + "=" * 80)

    print(
        "FINAL TEST RESULTS: "
        "MEAN ± SAMPLE STD OVER SEEDS "
        + ", ".join(
            str(seed)
            for seed in SEEDS
        )
    )

    print("=" * 80)

    for metric_name in preferred_metric_order:
        if metric_name not in summary:
            continue

        mean_value = (
            summary[metric_name]["mean"]
        )

        std_value = (
            summary[metric_name]["std"]
        )

        print(
            f"{metric_name:25s}: "
            f"{mean_value:.4f} ± "
            f"{std_value:.4f}"
        )


    print("\nResults saved under:")

    print(
        RESULT_ROOT
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    register_all_modules(
        init_default_scope=True
    )

    classes = validate_dataset_paths()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("\n" + "=" * 80)

    print(
        "EXPERIMENT CONFIGURATION"
    )

    print("=" * 80)

    print(
        f"Model config: {BASE_CONFIG}"
    )

    print(
        f"Pretrained checkpoint: "
        f"{PRETRAINED_CHECKPOINT}"
    )

    print(
        f"Training images: "
        f"{TRAIN_IMAGE_DIR}"
    )

    print(
        f"Training annotations: "
        f"{TRAIN_ANN_FILE}"
    )

    print(
        f"Validation images: "
        f"{VAL_IMAGE_DIR}"
    )

    print(
        f"Validation annotations: "
        f"{VAL_ANN_FILE}"
    )

    print(
        f"Test images: "
        f"{TEST_IMAGE_DIR}"
    )

    print(
        f"Test annotations: "
        f"{TEST_ANN_FILE}"
    )

    print(
        f"Classes: {classes}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Image scale: {IMAGE_SCALE}"
    )

    print(
        f"Epochs: {MAX_EPOCHS}"
    )

    print(
        f"Training batch size: "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        "Best checkpoint criterion: "
        "validation coco/segm_mAP"
    )

    print(
        f"Result directory: "
        f"{RESULT_ROOT}"
    )


    results = []

    for seed in SEEDS:
        result = train_and_test_seed(
            seed=seed,
            classes=classes,
        )

        results.append(
            result
        )


    write_final_summary(
        results
    )


if __name__ == "__main__":
    main()
