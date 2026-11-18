from .lib import *
from .load_background import load_background
from .select_random_object import select_random_object
from .extract_object import extract_object
from .place_object import place_object
from .update_existing_shapes_with_new_contour import update_existing_shapes_with_new_contour
from .apply_rotation_augmentation import apply_rotation_augmentation
from .apply_scale_augmentation import apply_scale_augmentation
from .apply_truncate_augmentation import apply_truncate_augmentation
from .prepare_masks import prepare_masks
from .shadow import ShadowSynthesizer, apply_shadow


def main_pipeline(
    object_images_folder,
    object_annotations_folder,
    background_image_path,
    background_json_path,
    output_folder_img,
    output_folder_annotations,
    num_objects,
    kernel_size,
    scale_factors,
    allowed_crop_directions,
    allowed_transforms,
    angles,
    only_complete_objects=False,
    mode='both',
    output_folder_shadow=None,
    save_instance_map=False,
    save_metadata=False,
    save_shadow_map=False,
    shadow_config=None,
    max_darken=0.6,
    reject_tiny_objects=False,
    use_blending=True,

    # --------------------------------------------------------
    # NEW: failure-aware copy-paste options
    # --------------------------------------------------------
    add_fn=False,
    fn_images_folder=None,
    fn_annotations_folder=None,
    fn_number_per_image=0,

    add_distractor=False,
    distractor_images_folder=None,
    distractor_annotations_folder=None,
    distractor_number_per_image=0,

    normal_allowed_labels=None,
    fn_allowed_labels=None,
    distractor_allowed_labels=None,
    add_distractor_label_to_output_json=False,

    # Source-specific quality/completeness filters.
    # Normal objects usually should be clean/complete.
    # FN hard examples and FP distractors usually should keep tiny/partial regions.
    fn_only_complete_objects=False,
    fn_reject_tiny_objects=False,
    distractor_only_complete_objects=False,
    distractor_reject_tiny_objects=False,

    # Keep FN/FP transformations conservative by default.
    # Hard examples are already hard; too much augmentation can make them noisy.
    extra_allowed_transforms=None,
    extra_scale_range=(0.97, 1.05),
    extra_angle_range=(-3.0, 3.0),

    # Independent per-image RNG streams. These keep shared components identical
    # when FN, FP, or shadow is enabled/disabled in an ablation.
    normal_rng=None,
    fn_rng=None,
    distractor_rng=None,
):
    """
    mode:
        'synthetic' -> save only the copy-paste image (no shadow).
        'shadow'    -> save only the shadow-augmented image.
        'both'      -> save the plain copy-paste image AND the shadowed image.

    NEW flags:
        add_fn=True:
            Selects missed GT objects from fn_images_folder + fn_annotations_folder
            and pastes them with their original source labels.

        add_distractor=True:
            Selects false-positive masks from distractor_images_folder +
            distractor_annotations_folder and pastes them visually, but does NOT
            append their label to the output JSON. Existing annotations are
            still trimmed if the distractor occludes them.

    Important:
        Distractor should NOT be a YOLO class later. It is only a temporary
        LabelMe label in the mined source JSONs so the pipeline can extract it.
    """

    if mode not in ('synthetic', 'shadow', 'both'):
        raise ValueError("mode must be one of: 'synthetic', 'shadow', 'both'")

    # Backward-compatible defaults. The deterministic driver passes dedicated
    # random.Random instances here.
    normal_rng = normal_rng if normal_rng is not None else random
    fn_rng = fn_rng if fn_rng is not None else random
    distractor_rng = distractor_rng if distractor_rng is not None else random

    if add_fn and (not fn_images_folder or not fn_annotations_folder):
        raise ValueError("add_fn=True requires fn_images_folder and fn_annotations_folder")

    if add_distractor and (not distractor_images_folder or not distractor_annotations_folder):
        raise ValueError("add_distractor=True requires distractor_images_folder and distractor_annotations_folder")

    if extra_allowed_transforms is None:
        # For hard examples and distractors: mostly keep native appearance.
        extra_allowed_transforms = (['none'] * 90 + ['scale'] * 10)

    (initial_background_image, modified_background_image, modified_background_data) = load_background(
        background_image_path,
        background_json_path
    )

    background_height, background_width, _ = initial_background_image.shape

    # ========================================================
    # SHADOW-SUPPORT OUTPUTS
    # instance_id_map: 0 = background, k = kth pasted visible region.
    # Distractors are included in this map so shadows remain consistent with
    # the generated pixels, although they are not added as training labels.
    # ========================================================
    instance_id_map = np.zeros((background_height, background_width), dtype=np.uint16)
    objects_metadata = []
    placed_index = 0

    def _sample_extra_transform(rng):
        transform_type = rng.choice(extra_allowed_transforms)

        angle = None
        scale_factor = None

        if transform_type == 'rotate':
            angle = rng.uniform(extra_angle_range[0], extra_angle_range[1])
        elif transform_type == 'scale':
            scale_factor = rng.uniform(extra_scale_range[0], extra_scale_range[1])

        return transform_type, angle, scale_factor

    def _paste_one_region(
        source_images_folder,
        source_annotations_folder,
        allowed_labels,
        transform_type,
        angle,
        scale_factor,
        source_kind,
        append_label=True,
        output_label=None,
        source_only_complete_objects=False,
        source_reject_tiny_objects=False,
        max_select_attempts=100,
        rng=None,
    ):
        """Select, transform, paste one source region, and update annotations.

        Returns True if one region was pasted, False otherwise.
        """

        nonlocal modified_background_image, modified_background_data, placed_index

        if rng is None:
            rng = random

        x_offset = None
        y_offset = None

        for _ in range(max_select_attempts):
            try:
                annotated_image, selected_shape, object_image_name = select_random_object(
                    source_images_folder,
                    source_annotations_folder,
                    only_complete_objects=source_only_complete_objects,
                    reject_tiny_objects=source_reject_tiny_objects,
                    allowed_labels=allowed_labels,
                    rng=rng,
                )
            except Exception as e:
                print(f"Could not select {source_kind}: {e}")
                return False

            object_name, contour_points, object_extracted, x, y, w, h = extract_object(
                annotated_image,
                selected_shape
            )

            if object_extracted is None or object_extracted.size == 0:
                continue

            try:
                if transform_type == 'rotate':
                    contour_points, mask_image, binary_mask_single_channel_crop, blurred_edge_region, outer_region, inner_region = \
                        apply_rotation_augmentation(
                            object_extracted, contour_points, angle, kernel_size,
                            use_blending=use_blending)

                elif transform_type == 'scale':
                    contour_points, mask_image, binary_mask_single_channel_crop, blurred_edge_region, outer_region, inner_region = \
                        apply_scale_augmentation(
                            object_extracted, contour_points, kernel_size, scale_factor,
                            use_blending=use_blending)

                elif transform_type == 'Truncate':
                    crop_direction = rng.choice(allowed_crop_directions)
                    contour_points, mask_image, binary_mask_single_channel_crop, blurred_edge_region, outer_region, inner_region, x_offset, y_offset = \
                        apply_truncate_augmentation(
                            object_extracted,
                            contour_points,
                            x,
                            y,
                            w,
                            h,
                            background_width,
                            background_height,
                            kernel_size,
                            crop_direction,
                            rng=rng,
                            use_blending=use_blending,
                        )
                else:
                    contour_points, mask_image, binary_mask_single_channel_crop, blurred_edge_region, outer_region, inner_region = \
                        prepare_masks(
                            object_extracted, contour_points, kernel_size,
                            use_blending=use_blending)

            except Exception as e:
                print(f"Skipping {source_kind} due to transform error: {e}")
                continue

            if binary_mask_single_channel_crop.shape[:2] != mask_image.shape[:2]:
                raise ValueError("Both mask images should have the same dimensions.")

            output_image = np.zeros((mask_image.shape[0], mask_image.shape[1], 4), dtype=np.uint8)
            output_image[:, :, :3] = mask_image

            object_height, object_width, _ = output_image.shape

            if object_height <= background_height and object_width <= background_width:
                break

            print(f"Selected {source_kind} does not fit, selecting another source region...")
        else:
            print(f"Could not place any {source_kind} after {max_select_attempts} attempts.")
            return False

        modified_background_image, adjusted_contour, x_offset, y_offset = place_object(
            transform_type,
            contour_points,
            background_json_path,
            background_width,
            background_height,
            object_width,
            object_height,
            binary_mask_single_channel_crop,
            modified_background_image,
            output_image,
            blurred_edge_region,
            outer_region,
            inner_region,
            kernel_size,
            x_offset,
            y_offset,
            rng=rng,
            use_blending=use_blending,
        )

        if adjusted_contour is None:
            return False

        # ====================================================
        # RECORD FULL MASK + Z-ORDER FOR SHADOWS
        # ====================================================
        placed_index += 1

        map_region = instance_id_map[y_offset:y_offset + object_height,
                                     x_offset:x_offset + object_width]

        if map_region.shape == binary_mask_single_channel_crop.shape[:2]:
            map_region[binary_mask_single_channel_crop > 127] = placed_index

        final_label = output_label if output_label is not None else object_name

        objects_metadata.append({
            'index': placed_index,
            'source_kind': source_kind,
            'source_label': object_name,
            'label_added_to_json': final_label if append_label else None,
            'source_image': object_image_name,
            'transform': transform_type,
            'rotation': angle,
            'scale_factor': scale_factor,
            'object_width': int(object_width),
            'object_height': int(object_height),
            'offset': [int(x_offset), int(y_offset)],
            'bbox': [int(x_offset), int(y_offset), int(object_width), int(object_height)],
            'points': adjusted_contour.tolist(),
        })

        # ====================================================
        # UPDATE EXISTING ANNOTATIONS
        # This is important for BOTH labeled objects and unlabeled distractors:
        # if the pasted region occludes a real object, the real object's polygon
        # must be trimmed.
        # ====================================================
        modified_background_data['shapes'] = update_existing_shapes_with_new_contour(
            modified_background_data['shapes'],
            adjusted_contour,
        )

        # ====================================================
        # APPEND NEW LABEL ONLY FOR OBJECTS/FN
        # For distractors, append_label=False, so they appear in the image but
        # not in the training labels.
        # ====================================================
        if append_label:
            modified_background_data['shapes'].append({
                'label': final_label,
                'points': adjusted_contour.tolist(),
                'group_id': None,
                'shape_type': 'polygon',
                'flags': {},
            })

        return True

    # ========================================================
    # 1) NORMAL COPY-PASTE OBJECTS
    # ========================================================
    for i in range(num_objects):
        transform_type = normal_rng.choice(allowed_transforms)
        angle = angles[i] if transform_type == 'rotate' else None
        scale_factor = scale_factors[i] if transform_type == 'scale' else None

        _paste_one_region(
            source_images_folder=object_images_folder,
            source_annotations_folder=object_annotations_folder,
            allowed_labels=normal_allowed_labels,
            transform_type=transform_type,
            angle=angle,
            scale_factor=scale_factor,
            source_kind='normal_object',
            append_label=True,
            output_label=None,   # read label from source JSON
            source_only_complete_objects=only_complete_objects,
            source_reject_tiny_objects=reject_tiny_objects,
            rng=normal_rng,
        )

    # ========================================================
    # 2) FN HARD OBJECTS: paste with Object label
    # ========================================================
    if add_fn and fn_number_per_image > 0:
        for _ in range(fn_number_per_image):
            transform_type, angle, scale_factor = _sample_extra_transform(fn_rng)

            _paste_one_region(
                source_images_folder=fn_images_folder,
                source_annotations_folder=fn_annotations_folder,
                allowed_labels=fn_allowed_labels,
                transform_type=transform_type,
                angle=angle,
                scale_factor=scale_factor,
                source_kind='false_negative_hard_object',
                append_label=True,
                output_label=None,   # read label from json_fn_only
                source_only_complete_objects=fn_only_complete_objects,
                source_reject_tiny_objects=fn_reject_tiny_objects,
                rng=fn_rng,
            )

    # ========================================================
    # 3) FP DISTRACTORS: paste pixels, update occluded labels, but do NOT
    #    append a new training label
    # ========================================================
    if add_distractor and distractor_number_per_image > 0:
        for _ in range(distractor_number_per_image):
            transform_type, angle, scale_factor = _sample_extra_transform(distractor_rng)

            _paste_one_region(
                source_images_folder=distractor_images_folder,
                source_annotations_folder=distractor_annotations_folder,
                allowed_labels=distractor_allowed_labels,
                transform_type=transform_type,
                angle=angle,
                scale_factor=scale_factor,
                source_kind='false_positive_distractor',
                append_label=add_distractor_label_to_output_json,
                output_label=None,   # if saved for debug, read label from json_fp_only
                source_only_complete_objects=distractor_only_complete_objects,
                source_reject_tiny_objects=distractor_reject_tiny_objects,
                rng=distractor_rng,
            )

    # ========================================================
    # SAVE
    # ========================================================
    background_name = os.path.splitext(os.path.basename(background_image_path))[0]

    # Keep LabelMe metadata synchronized with the actual generated image.
    # This is especially important when source/background image dimensions differ
    # or the original image extension is not JPG.
    modified_background_data['imagePath'] = f'{background_name}.jpg'
    modified_background_data['imageHeight'] = int(background_height)
    modified_background_data['imageWidth'] = int(background_width)
    # The generated image is stored externally, so stale embedded image data from
    # the source JSON should not be carried over.
    if 'imageData' in modified_background_data:
        modified_background_data['imageData'] = None

    os.makedirs(output_folder_annotations, exist_ok=True)
    output_annotations_path = os.path.join(output_folder_annotations, f'{background_name}.json')
    with open(output_annotations_path, 'w') as output_annotations_file:
        json.dump(modified_background_data, output_annotations_file, indent=2)

    if mode in ('synthetic', 'both'):
        os.makedirs(output_folder_img, exist_ok=True)
        output_image_path = os.path.join(output_folder_img, f'{background_name}.jpg')
        cv2.imwrite(output_image_path, modified_background_image)

    # Shadow generation
    shadow = None
    need_shadow = mode in ('shadow', 'both') or save_shadow_map
    if need_shadow:
        synth = ShadowSynthesizer(**(shadow_config or {}))
        shadow = synth(instance_id_map)

    if mode in ('shadow', 'both'):
        if output_folder_shadow is None:
            raise ValueError(f"output_folder_shadow must be set for mode '{mode}'.")
        os.makedirs(output_folder_shadow, exist_ok=True)
        shadowed_image = apply_shadow(modified_background_image, shadow, max_darken=max_darken)
        shadow_image_path = os.path.join(output_folder_shadow, f'{background_name}.jpg')
        cv2.imwrite(shadow_image_path, shadowed_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Optional support outputs
    if save_instance_map:
        instance_map_folder = os.path.join(os.path.dirname(output_folder_annotations), 'instance_maps')
        os.makedirs(instance_map_folder, exist_ok=True)
        instance_map_path = os.path.join(instance_map_folder, f'{background_name}.png')
        cv2.imwrite(instance_map_path, instance_id_map)

    if save_shadow_map and shadow is not None:
        shadow_map_folder = os.path.join(os.path.dirname(output_folder_annotations), 'shadow_maps')
        os.makedirs(shadow_map_folder, exist_ok=True)
        shadow_map_path = os.path.join(shadow_map_folder, f'{background_name}.png')
        cv2.imwrite(shadow_map_path, (shadow * 255).astype(np.uint8))

    if save_metadata:
        metadata_folder = os.path.join(os.path.dirname(output_folder_annotations), 'metadata')
        os.makedirs(metadata_folder, exist_ok=True)
        metadata_path = os.path.join(metadata_folder, f'{background_name}.json')
        with open(metadata_path, 'w') as metadata_file:
            json.dump({
                'image': f'{background_name}.jpg',
                'width': int(background_width),
                'height': int(background_height),
                'instance_map': f'{background_name}.png',
                'use_blending': bool(use_blending),
                'blending_method': 'ring_based_gaussian' if use_blending else 'hard_copy_paste',
                'objects': objects_metadata,
            }, metadata_file, indent=2)
