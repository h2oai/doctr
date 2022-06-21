# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Dict, List

import numpy as np
from scipy import io as sio
from tqdm import tqdm

from doctr.datasets.utils import crop_bboxes_from_image

import json
import cv2

np_dtype = np.float32

# Load mat data
tmp_root = 'SynthText/SynthText'
jpg_dir_name = 'SynthText_Reco_train'
target_path = "SynthText_prepped"
jpg_path = os.path.join(tmp_root, jpg_dir_name)

mat_data = sio.loadmat(os.path.join(tmp_root, 'gt.mat'))
train_samples = int(len(mat_data['imnames'][0]) * 0.9)
set_slice = slice(train_samples)
paths = mat_data['imnames'][0][set_slice]
boxes = mat_data['wordBB'][0][set_slice]
labels = mat_data['txt'][0][set_slice]
del mat_data

index = 0
labels_dict = {}
for img_path, word_boxes, txt in tqdm(iterable=zip(paths, boxes, labels),
                                        desc='Unpacking SynthText', total=len(paths)):
    # File existence check
    if not os.path.exists(os.path.join(tmp_root, img_path[0])):
        raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path[0])}")

    labels = [elt for word in txt.tolist() for elt in word.split()]
    # (x, y) coordinates of top left, top right, bottom right, bottom left corners
    word_boxes = word_boxes.transpose(2, 1, 0) if word_boxes.ndim == 3 else np.expand_dims(
        word_boxes.transpose(1, 0), axis=0)

    word_boxes = np.concatenate((word_boxes.min(axis=1), word_boxes.max(axis=1)), axis=1)

    os.makedirs(target_path + "/images", exist_ok = True)

    crops = crop_bboxes_from_image(img_path=os.path.join(tmp_root, img_path[0]), geoms=word_boxes)
    for crop, label in zip(crops, labels):
        if (crop.shape[0] == 0) or (crop.shape[1] == 0):
            # print(f"skipping crop because it has an invalid shape {crop.shape}")
            continue
        cv2.imwrite(f"{target_path}/images/{str(index)}.jpg", crop)
        labels_dict[f"{str(index)}.jpg"] = label
        index += 1

with open(target_path + "/labels.json", 'w') as json_file:
    json.dump(labels_dict, json_file)


