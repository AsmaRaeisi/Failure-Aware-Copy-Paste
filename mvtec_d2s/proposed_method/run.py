
import os
import json
from copy import deepcopy

from CP.main_pipeline import main_pipeline
from CP.random_utils import make_rng, stable_seed


# ============================================================
# Update the data root paths (lines 14–24) and the experiment settings (lines 82–93).
# ============================================================

object_images_folder = '**/train/images/'
object_annotations_folder = '**/train/jsons/'

background_images_folder = '**/train/images/'
background_annotations_folder = '**/train/jsons/'

hard_example_images_folder = '**/mined/images'
fn_annotations_folder = '**/mined/json_fn_only'
distractor_annotations_folder = '**/json_fp_only_distractor'

OUTPUT_ROOT = '**/experiments/'


# ONE immutable class-ID definition for the whole project.
# Keep using this same file for real train, val, test, and all synthetic runs.
CLASS_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classes.json')


def _load_shared_class_names(class_map_path):
    if not os.path.isfile(class_map_path):
        raise FileNotFoundError(
            f'Shared class map not found: {class_map_path}'
        )

    with open(class_map_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    names = data.get('names')
    if not isinstance(names, list) or not names:
        raise ValueError(
            f'Invalid classes.json: expected a non-empty "names" list: '
            f'{class_map_path}'
        )

    names = tuple(str(name).strip() for name in names)
    if len(names) != len(set(names)):
        raise ValueError('Invalid classes.json: duplicate class names found.')

    expected = {label: idx for idx, label in enumerate(names)}
    if 'label_to_id' in data and data['label_to_id'] != expected:
        raise ValueError(
            'Invalid classes.json: names and label_to_id are inconsistent.'
        )

    return names


# This is also the whitelist used when selecting labeled normal/FN objects.
# The selected source object still keeps its own original label; this only
# ensures that every labeled synthetic object belongs to the shared class map.
SHARED_CLASS_LABELS = _load_shared_class_names(CLASS_MAP_PATH)


# ============================================================
# ABLATION TABLE
# ============================================================

# Each experiment controls its own counts and blending setting.
# Set USE_BLENDING to True for ring-based Gaussian blending, or False
# for hard copy-paste. Use unique names for every ablation row.
#
# normal_count:
#   - integer, e.g. 3, for an exact number of normal inserted objects
#   - (minimum, maximum), e.g. (1, 4), for a deterministic inclusive range
# fn_count / fp_count:
#   - non-negative integers
#


# Update this to produce the experiments you want.
EXPERIMENTS = {
    'exp60': {'normal_count': (2, 8), 'fn_count': (2,8), 'fp_count': 0,'shadow': True, 'USE_BLENDING': True,},


}

# Update this too

EXPERIMENTS_TO_RUN = [
    'exp60',
]



MASTER_SEED = 42


# ============================================================
# QUALITY / AUGMENTATION SETTINGS
# ============================================================

NORMAL_ONLY_COMPLETE_OBJECTS = True
# Dataset-generic default: do not reject classes using ARMBench-specific absolute pixel thresholds.
NORMAL_REJECT_TINY_OBJECTS = False
FN_ONLY_COMPLETE_OBJECTS = False
FN_REJECT_TINY_OBJECTS = False
DISTRACTOR_ONLY_COMPLETE_OBJECTS = False
DISTRACTOR_REJECT_TINY_OBJECTS = False

# Labeled synthetic objects are selected only from the classes declared in
# classes.json. Their original source labels are preserved after pasting.
NORMAL_ALLOWED_LABELS = SHARED_CLASS_LABELS
FN_ALLOWED_LABELS = SHARED_CLASS_LABELS
DISTRACTOR_ALLOWED_LABELS = ('Distractor',)
ADD_DISTRACTOR_LABEL_TO_OUTPUT_JSON = False

EXTRA_ALLOWED_TRANSFORMS = ['none'] * 90 + ['scale'] * 10
EXTRA_SCALE_RANGE = (0.97, 1.05)
EXTRA_ANGLE_RANGE = (-3.0, 3.0)

SAVE_INSTANCE_MAP = False
# Keep metadata on: it lets you verify that paired rows used identical sources,
# transforms, and offsets for their shared components.
SAVE_METADATA = True
SAVE_SHADOW_MAP = False
SAVE_SHADOW_MAP_LIMIT = 0

# False: save only LabelMe JSON + generated images.
# True: after each selected experiment finishes, run the included shared-class
# labelme2yolo.py converter in instance-segmentation mode using CLASS_MAP_PATH.
# Its output is written
# under <experiment>/json/YOLODataset_seg/.
CONVERT_OUTPUT_TO_YOLO = True

MAX_DARKEN = 0.6
BASE_SHADOW_CONFIG = {
    'shadow_dir': (0.5, 1.0),
    'offset_scale': 0.12,
    'penumbra': 0.04,
    'strength': 0.60,
    'contact': 0.85,
    'jitter': 0.15,
    'cast_on_objects': True,
}


def _resolve_count(count_spec, rng, field_name):
    """Resolve an exact count or a deterministic inclusive count range."""
    if isinstance(count_spec, int):
        if count_spec < 0:
            raise ValueError(f'{field_name} must be non-negative, got {count_spec}')
        return count_spec

    if (
        isinstance(count_spec, (tuple, list))
        and len(count_spec) == 2
        and all(isinstance(value, int) for value in count_spec)
    ):
        minimum, maximum = count_spec
        if minimum < 0 or maximum < minimum:
            raise ValueError(
                f'{field_name} range must satisfy 0 <= minimum <= maximum, '
                f'got {count_spec}'
            )
        return rng.randint(minimum, maximum)

    raise TypeError(
        f'{field_name} must be a non-negative integer or a two-integer range, '
        f'got {count_spec!r}'
    )


def _convert_to_yolo(output_folder_annotations, output_folder_img, class_map_path):
    """Convert synthetic LabelMe annotations using the shared fixed IDs."""
    try:
        from labelme2yolo import Labelme2YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'YOLO conversion is enabled, but a converter dependency is missing. '
            'Install the dependencies described in README_LABELME2YOLO.md, including labelme.'
        ) from exc

    converter = Labelme2YOLO(
        output_folder_annotations,
        output_folder_img,
        class_map_path,
        to_seg=True,
    )
    converter.convert()


def generate_experiment(experiment_name):
    cfg = EXPERIMENTS[experiment_name]
    mode = 'shadow' if cfg['shadow'] else 'synthetic'
    use_blending = cfg['USE_BLENDING']

    # Each uniquely named experiment writes to its own output directory.
    experiment_root = os.path.join(OUTPUT_ROOT, experiment_name)
    # All final training images are written to <experiment>/img, whether or not
    # shadow is enabled. This makes the dataset paths consistent across rows.
    output_folder_img = os.path.join(experiment_root, 'img')
    output_folder_annotations = os.path.join(experiment_root, 'json')
    output_folder_shadow = output_folder_img

    os.makedirs(output_folder_img, exist_ok=True)
    os.makedirs(output_folder_annotations, exist_ok=True)

    image_index = 0

    for background_image_name in sorted(os.listdir(background_images_folder)):
        if not background_image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_index += 1
        background_stem = os.path.splitext(background_image_name)[0]
        background_image_path = os.path.join(background_images_folder, background_image_name)
        background_json_path = os.path.join(background_annotations_folder, f'{background_stem}.json')

        if not os.path.exists(background_json_path):
            print(f'Skipping {background_image_name}: missing background JSON')
            continue

        output_image_path = os.path.join(output_folder_img, f'{background_stem}.jpg')
        output_json_path = os.path.join(output_folder_annotations, f'{background_stem}.json')

        # Because every image has its own component RNGs, skipping an already
        # completed image does not change any later image.
        if os.path.exists(output_image_path) and os.path.exists(output_json_path):
            print(f'[{experiment_name}] Skipping {background_image_name}')
            continue

        normal_rng = make_rng(MASTER_SEED, background_image_name, 'normal')
        fn_rng = make_rng(MASTER_SEED, background_image_name, 'fn')
        fp_rng = make_rng(MASTER_SEED, background_image_name, 'fp')

        # IMPORTANT: preserve the exact legacy exp29 random sequence.
        # The ranged path is unchanged and is still used by exp40--exp43.
        normal_count_spec = cfg['normal_count']

        if isinstance(normal_count_spec, int):
            # Fixed-count ablations (exp44/exp45) must start from the same
            # normal-object RNG state as v4. First reproduce v4's random count
            # and its angle/scale draws. The requested fixed count then uses a
            # prefix of that legacy object set; if more objects are requested,
            # only their extra angle/scale parameters use an independent stream.
            legacy_count = normal_rng.randint(1, 4)
            legacy_angles = [
                normal_rng.uniform(-3, 3)
                for _ in range(legacy_count)
            ]

            legacy_scale_factors = []
            for _ in range(legacy_count):
                r = normal_rng.random()
                if r < 0.15:
                    legacy_scale_factors.append(normal_rng.uniform(0.90, 0.97))
                elif r < 0.85:
                    legacy_scale_factors.append(normal_rng.uniform(0.97, 1.05))
                else:
                    legacy_scale_factors.append(normal_rng.uniform(1.05, 1.12))

            num_objects_per_image = _resolve_count(
                normal_count_spec,
                normal_rng,
                'normal_count',
            )
            angles = legacy_angles[:num_objects_per_image]
            scale_factors = legacy_scale_factors[:num_objects_per_image]

            extra_count = num_objects_per_image - legacy_count
            if extra_count > 0:
                extra_param_rng = make_rng(
                    MASTER_SEED,
                    background_image_name,
                    'normal_fixed_count_extra_parameters',
                )
                angles.extend(
                    extra_param_rng.uniform(-3, 3)
                    for _ in range(extra_count)
                )
                for _ in range(extra_count):
                    r = extra_param_rng.random()
                    if r < 0.15:
                        scale_factors.append(extra_param_rng.uniform(0.90, 0.97))
                    elif r < 0.85:
                        scale_factors.append(extra_param_rng.uniform(0.97, 1.05))
                    else:
                        scale_factors.append(extra_param_rng.uniform(1.05, 1.12))
        else:
            # Do not change this path: it is the exact v4 sequence used by
            # exp40--exp43 when normal_count=(1, 4).
            num_objects_per_image = _resolve_count(
                normal_count_spec,
                normal_rng,
                'normal_count',
            )
            angles = [
                normal_rng.uniform(-3, 3)
                for _ in range(num_objects_per_image)
            ]

            scale_factors = []
            for _ in range(num_objects_per_image):
                r = normal_rng.random()
                if r < 0.15:
                    scale_factors.append(normal_rng.uniform(0.90, 0.97))
                elif r < 0.85:
                    scale_factors.append(normal_rng.uniform(0.97, 1.05))
                else:
                    scale_factors.append(normal_rng.uniform(1.05, 1.12))

        fn_number_per_image = _resolve_count(cfg['fn_count'], fn_rng, 'fn_count')
        fp_number_per_image = _resolve_count(cfg['fp_count'], fp_rng, 'fp_count')

        allowed_transforms = ['none'] * 75 + ['scale'] * 15 + ['rotate'] * 10
        allowed_crop_directions = ['top', 'bottom', 'left', 'right']
        kernel_size = 5

        shadow_config = deepcopy(BASE_SHADOW_CONFIG)
        # Independent deterministic shadow seed for this image.
        shadow_config['seed'] = stable_seed(MASTER_SEED, background_image_name, 'shadow')

        save_shadow_map_this = SAVE_SHADOW_MAP and (
            SAVE_SHADOW_MAP_LIMIT is None or image_index <= SAVE_SHADOW_MAP_LIMIT
        )

        main_pipeline(
            object_images_folder,
            object_annotations_folder,
            background_image_path,
            background_json_path,
            output_folder_img,
            output_folder_annotations,
            num_objects_per_image,
            kernel_size,
            scale_factors,
            allowed_crop_directions,
            allowed_transforms,
            angles,
            only_complete_objects=NORMAL_ONLY_COMPLETE_OBJECTS,
            mode=mode,
            output_folder_shadow=output_folder_shadow,
            save_instance_map=SAVE_INSTANCE_MAP,
            save_metadata=SAVE_METADATA,
            save_shadow_map=save_shadow_map_this,
            shadow_config=shadow_config,
            max_darken=MAX_DARKEN,
            use_blending=use_blending,
            reject_tiny_objects=NORMAL_REJECT_TINY_OBJECTS,

            add_fn=fn_number_per_image > 0,
            fn_images_folder=hard_example_images_folder,
            fn_annotations_folder=fn_annotations_folder,
            fn_number_per_image=fn_number_per_image,

            add_distractor=fp_number_per_image > 0,
            distractor_images_folder=hard_example_images_folder,
            distractor_annotations_folder=distractor_annotations_folder,
            distractor_number_per_image=fp_number_per_image,

            normal_allowed_labels=NORMAL_ALLOWED_LABELS,
            fn_allowed_labels=FN_ALLOWED_LABELS,
            distractor_allowed_labels=DISTRACTOR_ALLOWED_LABELS,
            add_distractor_label_to_output_json=ADD_DISTRACTOR_LABEL_TO_OUTPUT_JSON,

            fn_only_complete_objects=FN_ONLY_COMPLETE_OBJECTS,
            fn_reject_tiny_objects=FN_REJECT_TINY_OBJECTS,
            distractor_only_complete_objects=DISTRACTOR_ONLY_COMPLETE_OBJECTS,
            distractor_reject_tiny_objects=DISTRACTOR_REJECT_TINY_OBJECTS,

            extra_allowed_transforms=EXTRA_ALLOWED_TRANSFORMS,
            extra_scale_range=EXTRA_SCALE_RANGE,
            extra_angle_range=EXTRA_ANGLE_RANGE,

            normal_rng=normal_rng,
            fn_rng=fn_rng,
            distractor_rng=fp_rng,
        )

    if CONVERT_OUTPUT_TO_YOLO:
        print(f'[{experiment_name}] Converting generated LabelMe annotations to YOLO segmentation format')
        _convert_to_yolo(
            output_folder_annotations,
            output_folder_img,
            class_map_path=CLASS_MAP_PATH,
        )


if __name__ == '__main__':
    unknown = [name for name in EXPERIMENTS_TO_RUN if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f'Unknown experiments: {unknown}')

    for name in EXPERIMENTS_TO_RUN:
        cfg = EXPERIMENTS[name]
        for required_key in ('normal_count', 'fn_count', 'fp_count', 'shadow', 'USE_BLENDING'):
            if required_key not in cfg:
                raise KeyError(f"Experiment '{name}' is missing '{required_key}'")
        if not isinstance(cfg['USE_BLENDING'], bool):
            raise TypeError(f"Experiment '{name}': USE_BLENDING must be True or False")

        # Validate count specifications before creating output folders.
        _resolve_count(cfg['normal_count'], make_rng(MASTER_SEED, name, 'validate_normal'), 'normal_count')
        _resolve_count(cfg['fn_count'], make_rng(MASTER_SEED, name, 'validate_fn'), 'fn_count')
        _resolve_count(cfg['fp_count'], make_rng(MASTER_SEED, name, 'validate_fp'), 'fp_count')

        blending_name = 'ring-based blending' if cfg['USE_BLENDING'] else 'hard copy-paste (no blending)'
        print(f'\n===== Generating {name}: {cfg} | {blending_name} =====')
        generate_experiment(name)
