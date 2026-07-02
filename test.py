import argparse
import os.path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

from dataloader.loader_base import LOADER, get_image_paths

parser = argparse.ArgumentParser(description='TEST')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov8s')
parser.add_argument('-o', '--obb', action='store_true', help='whether obb')
parser.add_argument('-p', '--project', type=str, help='which object trained', default=None)
parser.add_argument('--show', action='store_true', help='whether show')
parser.add_argument('--auto', action='store_true', help='whether auto labeling')
parser.add_argument('--test', action='store_true', help='whether measure F2-score')
parser.add_argument('--work', type=int, help='num of workers for multiprocessing', default=16)
parser.add_argument('--dirs', type=str, help='path to load image data')
args = parser.parse_args()

if args.obb:
    args.model += '-obb'

model = None


def get_model():
    global model
    if model is None:
        model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')
    return model


def collate_fn(batch):
    paths, rescale_factor, image = zip(*batch)
    image = np.asarray(image)
    image = (image[..., ::-1]).transpose(0, 3, 1, 2)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).float()
    image /= 255
    return paths, rescale_factor, image


def test_collate_fn(batch):
    paths, rescale_factor, labels, image = zip(*batch)
    image = np.asarray(image)
    image = (image[..., ::-1]).transpose(0, 3, 1, 2)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).float()
    image /= 255
    return paths, rescale_factor, labels, image


def annotate_label(path: str,
                   rescale_factor: Tuple[bool, float, float],
                   r):
    path = '.'.join(path.split('.')[:-1] + ['txt'])
    w_h, scale, pad = rescale_factor
    if r.obb is not None:
        obb = r.obb
        x = obb.xyxyxyxyn.cpu().numpy().reshape(-1, 8)
        x[:, w_h::2] *= scale
        x[:, w_h::2] -= pad

        with open(path, 'w', encoding='utf-8') as f:
            for cls, box in zip(obb.cls, x):
                x1, y1, x2, y2, x3, y3, x4, y4 = box.flatten()
                f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    int(cls.item()), x1, y1, x2, y2, x3, y3, x4, y4))
    else:
        hbb = r.boxes
        x = hbb.xywhn.cpu().numpy()
        x[:, 2 + w_h] *= scale
        x[:, int(w_h)] = 0.5 + (x[:, int(w_h)] - 0.5) * scale

        with open(path, 'w', encoding='utf-8') as f:
            for cls, box in zip(hbb.cls, x):
                cx, cy, w, h = box
                f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(int(cls), cx, cy, w, h))


def load_cfg():
    with open(os.path.join('cfg', 'datasets', args.project + '.yaml'), 'r') as f:
        return yaml.safe_load(f)


def hbb_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def hbb_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = hbb_to_xyxy(a)
    b = hbb_to_xyxy(b)
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def obb_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1, 2).astype(np.float32)
    b = b.reshape(-1, 2).astype(np.float32)
    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def box_iou(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) == 8 and len(target) == 8:
        return obb_iou(pred, target)
    return hbb_iou(pred, target)


def predictions_from_result(r, rescale_factor: Tuple[bool, float, float]) -> List[Tuple[int, np.ndarray]]:
    predictions = []
    w_h, scale, pad = rescale_factor
    if r.obb is not None:
        boxes = r.obb.xyxyxyxyn.cpu().numpy().reshape(-1, 8)
        boxes[:, w_h::2] *= scale
        boxes[:, w_h::2] -= pad
        classes = r.obb.cls.cpu().numpy()
    else:
        boxes = r.boxes.xywhn.cpu().numpy()
        boxes[:, 2 + w_h] *= scale
        boxes[:, int(w_h)] = 0.5 + (boxes[:, int(w_h)] - 0.5) * scale
        classes = r.boxes.cls.cpu().numpy()
    for cls, box in zip(classes, boxes):
        predictions.append((int(cls), box.astype(np.float32)))
    return predictions


def f2_score(tp: int, fp: int, fn: int) -> float:
    denominator = 5 * tp + 4 * fn + fp
    return 0.0 if denominator == 0 else 5 * tp / denominator


def update_counts(counts: Dict[int, Dict[str, int]],
                  predictions: List[Tuple[int, np.ndarray]],
                  targets: List[Tuple[int, np.ndarray]],
                  iou_threshold: float = 0.5):
    for cls, _ in predictions + targets:
        counts.setdefault(cls, {'tp': 0, 'fp': 0, 'fn': 0})

    for cls in sorted(counts):
        pred_boxes = [box for pred_cls, box in predictions if pred_cls == cls]
        target_boxes = [box for target_cls, box in targets if target_cls == cls]
        matched = set()

        for pred_box in pred_boxes:
            best_iou = 0.0
            best_i = None
            for i, target_box in enumerate(target_boxes):
                if i in matched:
                    continue
                iou = box_iou(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
            if best_i is not None and best_iou >= iou_threshold:
                counts[cls]['tp'] += 1
                matched.add(best_i)
            else:
                counts[cls]['fp'] += 1

        counts[cls]['fn'] += len(target_boxes) - len(matched)


def print_f2_table(counts: Dict[int, Dict[str, int]], cls2name: Dict[int, str]):
    rows = []
    total = {'tp': 0, 'fp': 0, 'fn': 0}
    for cls in sorted(counts):
        row = counts[cls]
        total['tp'] += row['tp']
        total['fp'] += row['fp']
        total['fn'] += row['fn']
        rows.append([str(cls), cls2name.get(cls, str(cls)), row['tp'], row['fp'], row['fn'],
                     f'{f2_score(row["tp"], row["fp"], row["fn"]):.4f}'])
    rows.append(['all', 'all', total['tp'], total['fp'], total['fn'],
                 f'{f2_score(total["tp"], total["fp"], total["fn"]):.4f}'])

    headers = ['class', 'name', 'TP', 'FP', 'FN', 'F2']
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]

    def format_row(row):
        return ' | '.join(str(value).ljust(width) for value, width in zip(row, widths))

    print(format_row(headers))
    print('-+-'.join('-' * width for width in widths))
    for row in rows:
        print(format_row(row))


if args.auto:
    datasets = LOADER(args)
    dataloader = DataLoader(datasets,
                            batch_size=args.work,
                            num_workers=min(30, args.work),
                            collate_fn=collate_fn
                            )

    executor = ThreadPoolExecutor()
    with tqdm(total=len(datasets), ncols=80) as progress:
        for paths, rescale_factor, tensor in dataloader:
            progress.update(len(tensor))
            results = get_model().predict(tensor, verbose=False)
            executor.map(lambda args: annotate_label(*args), zip(paths, rescale_factor, results))

if args.test:
    datasets = LOADER(args)
    dataloader = DataLoader(datasets,
                            batch_size=args.work,
                            num_workers=min(30, args.work),
                            collate_fn=test_collate_fn)
    counts = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in datasets.cls2name}
    with tqdm(total=len(datasets), ncols=80) as progress:
        for _, rescale_factor, labels, tensor in dataloader:
            progress.update(len(tensor))
            results = get_model().predict(tensor, verbose=False)
            for result, factor, target in zip(results, rescale_factor, labels):
                update_counts(counts, predictions_from_result(result, factor), list(target))

    print_f2_table(counts, datasets.cls2name)

if args.show:
    cfg = load_cfg()
    if args.dirs:
        img_dir = args.dirs
    else:
        img_dir = os.path.join(cfg['path'], cfg['test'])
    images = get_image_paths(img_dir)
    cls2name = cfg['names']
    results = get_model().predict(source=img_dir,
                                  stream=True,
                                  verbose=False)
    paused = False
    for r in tqdm(results, total=len(images), ncols=80):
        if r.obb is not None:
            obb = r.obb
            for cls, box in zip(obb.cls, obb.xyxyxyxy.cpu()):
                cv2.polylines(r.orig_img, [np.asarray(box, dtype=int)], True, (0, 255, 0), 2)
                x1, y1 = box[0].int()
                cv2.putText(r.orig_img, cls2name[cls.item()], (x1.item(), y1.item() - 5), 0, 1, (0, 255, 0), 2, 16)
        else:
            hbb = r.boxes
            for cls, box in zip(hbb.cls, hbb.xyxy):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(r.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(r.orig_img, cls2name[cls.item()], (x1, y1 - 5), 0, 1, (0, 255, 0), 2, 16)
        cv2.imshow('sample', r.orig_img)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == 27:
            break
        elif key == 32:  # Space key
            paused = not paused
