# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tqdm import tqdm

from doctr import datasets
from doctr.file_utils import is_tf_available
from doctr.models import ocr_predictor, db_resnet50
from doctr.utils.geometry import extract_crops, extract_rcrops
from doctr.utils.metrics import LocalizationConfusion, OCRMetric, TextMatch
from scipy.optimize import linear_sum_assignment

# Enable GPU growth if using TF
if is_tf_available():
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if any(gpu_devices):
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    import torch


def _pct(val):
    return "N/A" if val is None else f"{val:.2%}"

def box_iou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Computes the IoU between two sets of bounding boxes

    Args:
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)

    Returns:
        the IoU matrix of shape (N, M)
    """

    iou_mat: np.ndarray = np.zeros((boxes_1.shape[0], boxes_2.shape[0]), dtype=np.float32)

    if boxes_1.shape[0] > 0 and boxes_2.shape[0] > 0:
        l1, t1, r1, b1 = np.split(boxes_1, 4, axis=1)
        l2, t2, r2, b2 = np.split(boxes_2, 4, axis=1)

        left = np.maximum(l1, l2.T)
        top = np.maximum(t1, t2.T)
        right = np.minimum(r1, r2.T)
        bot = np.minimum(b1, b2.T)

        intersection = np.clip(right - left, 0, np.Inf) * np.clip(bot - top, 0, np.Inf)
        union = (r1 - l1) * (b1 - t1) + ((r2 - l2) * (b2 - t2)).T - intersection
        iou_mat = intersection / union

        # convert nan to 0 in iou_mat
        iou_mat[np.isnan(iou_mat)] = 0

    return iou_mat



def main(args):

    if not args.rotation:
        args.eval_straight = True

    # switch to customized models
    det_detector = db_resnet50(pretrained=False)
    det_detector.load_state_dict(torch.load("/home/mzhao/Data/work/DocAI/src/doctr/db_resnet50_receipt2_ga5_3rc_f3.pt",map_location='cpu'))

    predictor = ocr_predictor(
        det_arch=det_detector,
        # det_arch = args.detection,
        reco_arch=args.recognition,
        pretrained=True,
        reco_bs=args.batch_size,
        assume_straight_pages=not args.rotation
    )
    print('aspect_ratio: ', predictor.det_predictor.pre_processor.resize.preserve_aspect_ratio)
    # predictor.det_predictor.pre_processor.resize.preserve_aspect_ratio = True
    print(predictor.det_predictor.pre_processor.resize.size)
    predictor.det_predictor.pre_processor.resize.size = (1536,1536)

    if args.img_folder and args.label_file:
        testset = datasets.OCRDataset(
            img_folder=args.img_folder,
            label_file=args.label_file,
        )
        sets = [testset]
    else:
        train_set = datasets.__dict__[args.dataset](train=True, download=True, use_polygons=not args.eval_straight)
        val_set = datasets.__dict__[args.dataset](train=False, download=True, use_polygons=not args.eval_straight)
        sets = [train_set, val_set]
        # sets = [train_set]

    reco_metric = TextMatch()
    if args.mask_shape:
        det_metric = LocalizationConfusion(
            iou_thresh=args.iou,
            use_polygons=not args.eval_straight,
            mask_shape=(args.mask_shape, args.mask_shape)
        )
        e2e_metric = OCRMetric(
            iou_thresh=args.iou,
            use_polygons=not args.eval_straight,
            mask_shape=(args.mask_shape, args.mask_shape)
        )
    else:
        det_metric = LocalizationConfusion(iou_thresh=args.iou, use_polygons=not args.eval_straight)
        e2e_metric = OCRMetric(iou_thresh=args.iou, use_polygons=not args.eval_straight)

    sample_idx = 0
    extraction_fn = extract_crops if args.eval_straight else extract_rcrops

    bad_file_list = []
    for dataset in sets:
        for idx, (page, target) in tqdm(enumerate(dataset)):
            # GT
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            if args.img_folder and args.label_file:
                x, y, w, h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
                xmin, ymin = np.clip(x - w / 2, 0, 1), np.clip(y - h / 2, 0, 1)
                xmax, ymax = np.clip(x + w / 2, 0, 1), np.clip(y + h / 2, 0, 1)
                gt_boxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)

            # Forward
            if is_tf_available():
                out = predictor(page[None, ...])
                crops = extraction_fn(page, gt_boxes)
                reco_out = predictor.reco_predictor(crops)
            else:
                torch.cuda.set_device(0)
                predictor = predictor.cuda()
                with torch.no_grad():
                    out = predictor(page[None, ...])
                    # We directly crop on PyTorch tensors, which are in channels_first
                    crops = extraction_fn(page, gt_boxes, channels_last=False)
                    reco_out = predictor.reco_predictor(crops)

            if len(reco_out):
                reco_words, _ = zip(*reco_out)
            else:
                reco_words = []

            # Unpack preds
            pred_boxes = []
            pred_labels = []
            for _page in out.pages:
                height, width = _page.dimensions
                for block in _page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if not args.rotation:
                                (a, b), (c, d) = word.geometry
                            else:
                                [x1, y1], [x2, y2], [x3, y3], [x4, y4], = word.geometry
                            if gt_boxes.dtype == int:
                                if not args.rotation:
                                    pred_boxes.append([int(a * width), int(b * height),
                                                       int(c * width), int(d * height)])
                                else:
                                    if args.eval_straight:
                                        pred_boxes.append(
                                            [
                                                int(width * min(x1, x2, x3, x4)),
                                                int(height * min(y1, y2, y3, y4)),
                                                int(width * max(x1, x2, x3, x4)),
                                                int(height * max(y1, y2, y3, y4)),
                                            ]
                                        )
                                    else:
                                        pred_boxes.append(
                                            [
                                                [int(x1 * width), int(y1 * height)],
                                                [int(x2 * width), int(y2 * height)],
                                                [int(x3 * width), int(y3 * height)],
                                                [int(x4 * width), int(y4 * height)],
                                            ]
                                        )
                            else:
                                if not args.rotation:
                                    pred_boxes.append([a, b, c, d])
                                else:
                                    if args.eval_straight:
                                        pred_boxes.append(
                                            [
                                                min(x1, x2, x3, x4),
                                                min(y1, y2, y3, y4),
                                                max(x1, x2, x3, x4),
                                                max(y1, y2, y3, y4),
                                            ]
                                        )
                                    else:
                                        pred_boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                            pred_labels.append(word.value)

            # try output bad predictions
            iou_mat = box_iou(gt_boxes, np.asarray(pred_boxes))
            gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
            matches =  int((iou_mat[gt_indices, pred_indices] >= 0.5).sum())
            match_pet = matches/len(gt_boxes)

            # print(match_pet)
            if match_pet <0.5:
                print(match_pet)
                print(len(pred_boxes))
                # print(pred_boxes)
                print(len(gt_boxes))
                # print(gt_boxes)
                print(dataset.data[idx][0])
                # print(page)
                
                bad_file_list.append(dataset[idx][0])            

            
            # Update the metric
            det_metric.update(gt_boxes, np.asarray(pred_boxes))
            reco_metric.update(gt_labels, reco_words)
            e2e_metric.update(gt_boxes, np.asarray(pred_boxes), gt_labels, pred_labels)

            # Loop break
            sample_idx += 1
            if isinstance(args.samples, int) and args.samples == sample_idx:
                break
        if isinstance(args.samples, int) and args.samples == sample_idx:
            break

    # Unpack aggregated metrics
    print(f"Model Evaluation (model= {args.detection} + {args.recognition}, "
          f"dataset={'OCRDataset' if args.img_folder else args.dataset})")
    recall, precision, mean_iou = det_metric.summary()
    print(f"Text Detection - Recall: {_pct(recall)}, Precision: {_pct(precision)}, Mean IoU: {_pct(mean_iou)}")
    acc = reco_metric.summary()
    print(f"Text Recognition - Accuracy: {_pct(acc['raw'])} (unicase: {_pct(acc['unicase'])})")
    recall, precision, mean_iou = e2e_metric.summary()
    print(f"OCR - Recall: {_pct(recall['raw'])} (unicase: {_pct(recall['unicase'])}), "
          f"Precision: {_pct(precision['raw'])} (unicase: {_pct(precision['unicase'])}), Mean IoU: {_pct(mean_iou)}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('detection', type=str, help='Text detection model to use for analysis')
    parser.add_argument('recognition', type=str, help='Text recognition model to use for analysis')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold to match a pair of boxes')
    parser.add_argument('--dataset', type=str, default='FUNSD', help='choose a dataset: FUNSD, CORD')
    parser.add_argument('--img_folder', type=str, default=None, help='Only for local sets, path to images')
    parser.add_argument('--label_file', type=str, default=None, help='Only for local sets, path to labels')
    parser.add_argument('--rotation', dest='rotation', action='store_true', help='run rotated OCR + postprocessing')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size for recognition')
    parser.add_argument('--mask_shape', type=int, default=None, help='mask shape for mask iou (only for rotation)')
    parser.add_argument('--samples', type=int, default=None, help='evaluate only on the N first samples')
    parser.add_argument('--eval-straight', action='store_true',
                        help='evaluate on straight pages with straight bbox (to use the quick and light metric)')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
