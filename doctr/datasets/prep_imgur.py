# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from doctr.datasets.utils import convert_target_to_relative, crop_bboxes_from_image

img_folder = "./IMGUR5K-Handwriting-Dataset/dataset"
target_folder = "./IMGUR5K-Handwriting-Dataset/images"
img_names = os.listdir(img_folder)
train_samples = int(len(img_names) * 0.9)
set_slice = slice(train_samples)
label_path = "./IMGUR5K-Handwriting-Dataset/dataset_info/imgur5k_annotations.json"
with open(label_path) as f:
    annotation_file = json.load(f)

index = 0
os.makedirs("./IMGUR5K-Handwriting-Dataset/images", exist_ok = True)
labels_dict = {}
for img_name in tqdm(iterable=img_names[set_slice], desc='Unpacking IMGUR5K', total=len(img_names[set_slice])):
    img_path = Path(img_folder, img_name)
    img_id = img_name.split(".")[0]

    # some files have no annotations which are marked with only a dot in the 'word' key
    # ref: https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset/blob/main/README.md
    if img_id not in annotation_file['index_to_ann_map'].keys():
        continue
    ann_ids = annotation_file['index_to_ann_map'][img_id]
    annotations = [annotation_file['ann_id'][a_id] for a_id in ann_ids]

    labels = [ann['word'] for ann in annotations if ann['word'] != '.']
    # x_center, y_center, width, height, angle
    _boxes = [list(map(float, ann['bounding_box'].strip('[ ]').split(', ')))
                for ann in annotations if ann['word'] != '.']
    # (x, y) coordinates of top left, top right, bottom right, bottom left corners
    box_targets = [cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), box[4])) for box in _boxes]
    box_targets = [np.concatenate((points.min(0), points.max(0)), axis=-1) for points in box_targets]
    # filter images without boxes
    if len(box_targets) > 0:
        crops = crop_bboxes_from_image(img_path=os.path.join(img_folder,  img_name),
                                        geoms=np.asarray(box_targets, dtype=np.float32))
        for crop, label in zip(crops, labels):
            if (crop.shape[0] == 0) or (crop.shape[1] == 0):
                continue
            cv2.imwrite(f"{target_folder}/{str(index)}.jpg", crop)
            labels_dict[f"{str(index)}.jpg"] = label
            index += 1
with open("./IMGUR5K-Handwriting-Dataset/labels.json", 'w') as json_file:
    json.dump(labels_dict, json_file)
