#!/usr/bin/env python3
# Modified version:
# Uses one persistent classes.json so YOLO class IDs remain identical
# across independently converted train, val, test, and synthetic folders.

'''
python labelme2yolo_shared_classes.py \
    --json_dir "/path/train/jsons" \
    --image_dir "/path/train/images" \
    --class_map "/path/classes.json" \
    --seg

    
'''
'''
Created on Aug 18, 2021

@author: xiaosonh
'''
import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
import cv2
import PIL.Image
import shutil 
from sklearn.model_selection import train_test_split


class Labelme2YOLO(object):
    
    def __init__(self, json_dir, image_dir, class_map, to_seg=False):

        self._json_dir = json_dir
        self._image_dir = image_dir
        self._class_map = class_map

        # IMPORTANT:
        # Always load the SAME persistent classes.json.
        # IDs are never rebuilt from the current train/val/test folder.
        self._label_id_map = self._get_label_id_map(self._class_map)
        self._to_seg = to_seg

        i = 'YOLODataset'
        i += '_seg/' if to_seg else '/'
        self._save_path_pfx = os.path.join(self._json_dir, i)

    def _make_output_dir(self):
        self._label_dir_path = os.path.join(self._save_path_pfx, 'labels')
        self._image_dir_path = os.path.join(self._save_path_pfx, 'images')

        for path in [self._label_dir_path, self._image_dir_path]:

            if os.path.exists(path):
                shutil.rmtree(path)

            os.makedirs(path)    
                
    def _get_label_id_map(self, class_map_path):
        """
        Load one persistent class mapping from classes.json.

        Expected format:
        {
          "names": ["class_a", "class_b", ...],
          "label_to_id": {
            "class_a": 0,
            "class_b": 1,
            ...
          }
        }

        The 'names' list defines the YOLO IDs:
            names[0] -> class ID 0
            names[1] -> class ID 1
            ...

        Using the same classes.json for every run guarantees identical
        class IDs across train, val, test, and synthetic datasets.
        """

        if not os.path.isfile(class_map_path):
            raise FileNotFoundError(
                f'Could not find shared classes.json: {class_map_path}'
            )

        with open(class_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'names' not in data:
            raise ValueError(
                f'Invalid classes.json: missing "names" field: {class_map_path}'
            )

        names = data['names']

        if not isinstance(names, list) or len(names) == 0:
            raise ValueError(
                f'Invalid classes.json: "names" must be a non-empty list.'
            )

        if len(names) != len(set(names)):
            raise ValueError(
                f'Invalid classes.json: duplicate class names were found.'
            )

        label_id_map = OrderedDict(
            (label, idx)
            for idx, label in enumerate(names)
        )

        # If label_to_id is included, verify that it agrees with "names".
        if 'label_to_id' in data:
            expected = dict(label_id_map)
            if data['label_to_id'] != expected:
                raise ValueError(
                    'Invalid classes.json: "names" and "label_to_id" '
                    'contain inconsistent mappings.'
                )

        print(f'Loaded shared class mapping from: {class_map_path}')
        print(f'Number of classes: {len(label_id_map)}')
        for label, idx in label_id_map.items():
            print(f'  {idx}: {label}')

        return label_id_map

    def _validate_labels(self):
        """Fail if the current folder contains a class absent from classes.json."""

        unknown_labels = set()

        for file_name in os.listdir(self._json_dir):
            if not file_name.endswith('.json'):
                continue

            json_path = os.path.join(self._json_dir, file_name)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for shape in data.get('shapes', []):
                label = shape.get('label')

                if label not in self._label_id_map:
                    unknown_labels.add(label)

        if unknown_labels:
            raise ValueError(
                'The current annotation folder contains labels that are not '
                'present in the shared classes.json:\n  '
                + '\n  '.join(sorted(str(x) for x in unknown_labels))
                + '\nUse the correct shared classes.json before converting.'
            )
    
    def _train_test_split(self, folders, json_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json' \
                                for train_sample_name in os.listdir(train_folder) \
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]
            
            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json' \
                              for val_sample_name in os.listdir(val_folder) \
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]
            
            return train_json_names, val_json_names
        
        
        if val_size == 0:
            train_idxs = range(len(json_names))
            val_idxs = []
        else:
            train_idxs, val_idxs = train_test_split(range(len(json_names)), test_size=val_size)
            
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]
        
        return train_json_names, val_json_names
    
    def convert(self):

        # Ensure this folder contains only labels defined by the shared map.
        self._validate_labels()

        json_names = [
            file_name for file_name in os.listdir(self._json_dir)
            if file_name.endswith('.json')
        ]

        self._make_output_dir()

        for json_name in json_names:

            json_path = os.path.join(self._json_dir, json_name)
            json_data = json.load(open(json_path))

            print(f'Converting {json_name} ...')

            img_path = self._save_yolo_image(
                json_data,
                json_name,
                self._image_dir_path
            )

            yolo_obj_list = self._get_yolo_object_list(
                json_data,
                img_path
            )

            self._save_yolo_label(
                json_name,
                self._label_dir_path,
                '',
                yolo_obj_list
            )

        
        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()
                
    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))
        
        print('Converting %s ...' % json_name)
        
        img_path = self._save_yolo_image(json_data, json_name, 
                                         self._json_dir, '')
        
        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        self._save_yolo_label(json_name, self._json_dir, 
                              '', yolo_obj_list)
    
    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []
        
        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data['shapes']:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
            
            yolo_obj_list.append(yolo_obj)
            
        return yolo_obj_list
    
    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        obj_center_x, obj_center_y = shape['points'][0]

        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)

        if self._to_seg:
            retval = [label_id]

            n_part = radius / 10
            n_part = int(n_part) if n_part > 4 else 4
            n_part2 = n_part << 1

            pt_quad = [None for i in range(0, 4)]
            pt_quad[0] = [[obj_center_x + math.cos(i * math.pi / n_part2) * radius,
                         obj_center_y - math.sin(i * math.pi / n_part2) * radius]
                         for i in range(1, n_part)]
            pt_quad[1] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[0]]
            pt_quad[1].reverse()
            pt_quad[3] = [[x1, obj_center_y * 2 - y1] for x1, y1 in pt_quad[0]]
            pt_quad[3].reverse()
            pt_quad[2] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[3]]
            pt_quad[2].reverse()

            pt_quad[0].append([obj_center_x, obj_center_y - radius])
            pt_quad[1].append([obj_center_x - radius, obj_center_y])
            pt_quad[2].append([obj_center_x, obj_center_y + radius])
            pt_quad[3].append([obj_center_x + radius, obj_center_y])

            for i in pt_quad:
                for j in i:
                    j[0] = round(float(j[0]) / img_w, 6)
                    j[1] = round(float(j[1]) / img_h, 6)
                    retval.extend(j)
            return retval

        obj_w = 2 * radius
        obj_h = 2 * radius
        
        yolo_center_x= round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]

        if self._to_seg:
            retval = [label_id]
            for i in shape['points']:
                i[0] = round(float(i[0]) / img_w, 6)
                i[1] = round(float(i[1]) / img_h, 6)
                retval.extend(i)
            return retval

        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)
            
            x_lists = [port[0] for port in obj_port_list]        
            y_lists = [port[1] for port in obj_port_list]
            
            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)
        
        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])
                    
        yolo_center_x= round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
    
    def _save_yolo_label(self, json_name, label_dir_path, target_dir, yolo_obj_list):
        txt_path = os.path.join(label_dir_path, 
                                target_dir, 
                                json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = ""
                for i in yolo_obj:
                    yolo_obj_line += f'{i} '
                yolo_obj_line = yolo_obj_line[:-1]
                if yolo_obj_idx != len(yolo_obj_list) - 1:
                    yolo_obj_line += '\n'
                f.write(yolo_obj_line)

    def _save_yolo_image(self, json_data, json_name, image_dir_path):

        img_name = json_data['imagePath']

        original_img_path = os.path.join(
            self._image_dir,
            img_name
        )

        save_img_path = os.path.join(
            image_dir_path,
            img_name
        )

        if not os.path.exists(original_img_path):
            raise FileNotFoundError(
                f'Could not find image: {original_img_path}'
            )

        shutil.copy(original_img_path, save_img_path)

        return save_img_path
    
    def _save_dataset_yaml(self):

        yaml_path = os.path.join(
            self._save_path_pfx,
            'dataset.yaml'
        )

        with open(yaml_path, 'w+') as yaml_file:

            yaml_file.write(
                f"train: {self._image_dir_path}\n"
            )

            yaml_file.write(
                f"val: {self._image_dir_path}\n\n"
            )

            yaml_file.write(
                f"nc: {len(self._label_id_map)}\n\n"
            )

            names_str = ', '.join(
                [f"'{label}'" for label in self._label_id_map]
            )

            yaml_file.write(f'names: [{names_str}]')
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--json_dir',
        type=str,
        required=True
    )

    parser.add_argument(
        '--image_dir',
        type=str,
        required=True
    )

    parser.add_argument(
        '--class_map',
        type=str,
        required=True,
        help='Path to the shared classes.json. Use the SAME file for every split.'
    )

    parser.add_argument(
        '--seg',
        action='store_true'
    )

    args = parser.parse_args()

    convertor = Labelme2YOLO(
        args.json_dir,
        args.image_dir,
        args.class_map,
        to_seg=args.seg
    )

    convertor.convert()
