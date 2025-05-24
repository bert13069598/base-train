import argparse
import os.path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

from dataloader.loader_base import LOADER

parser = argparse.ArgumentParser(description='TEST')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov8s')
parser.add_argument('-o', '--obb', action='store_true', help='whether obb')
parser.add_argument('-p', '--project', type=str, help='which object trained', default=None)
parser.add_argument('--show', action='store_true', help='whether show')
parser.add_argument('--auto', action='store_true', help='whether auto labeling')
parser.add_argument('--work', type=int, help='num of workers for multiprocessing', default=16)

args = parser.parse_args()

if args.obb:
    args.model += '-obb'


def collate_fn(image):
    image = np.asarray(image)
    image = (image[..., ::-1]).transpose(0, 3, 1, 2)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).float()
    image /= 255
    return image


def annotate_label(path, r):
    # path = '.'.join(r.path.split('.')[:-1] + ['txt'])
    path = '.'.join(path.split('.')[:-1] + ['txt'])
    if r.obb is not None:
        obb = r.obb
        with open(path, 'w', encoding='utf-8') as f:
            for cls, box in zip(obb.cls, obb.xyxyxyxyn):
                x1, y1, x2, y2, x3, y3, x4, y4 = box.flatten().cpu().numpy()
                f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    int(cls.item()), x1, y1, x2, y2, x3, y3, x4, y4))
    else:
        hbb = r.boxes
        with open(path, 'w', encoding='utf-8') as f:
            for cls, box in zip(hbb.cls, hbb.xywhn):
                cx, cy, w, h = box.cpu().numpy()
                f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(int(cls), cx, cy, w, h))


model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')

if args.auto:
    datasets = LOADER(args)
    dataloader = DataLoader(datasets,
                            batch_size=args.work,
                            num_workers=min(30, args.work),
                            pin_memory=True,
                            # persistent_workers=True,
                            collate_fn=collate_fn
                            )

    executor = ThreadPoolExecutor()
    progress = tqdm(total=len(datasets), ncols=80)
    cnt = 0
    for tensor in dataloader:
        progress.update(len(tensor))
        results = model.predict(tensor, verbose=False)
        executor.map(lambda args: annotate_label(*args), zip(datasets.images[cnt:cnt + len(tensor)], results))
        cnt += len(tensor)

if args.show:
    with open(os.path.join('cfg', 'datasets', args.project + '.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    root_path = cfg['path']
    cls2name = cfg['names']

    results = model.predict(source=os.path.join(root_path, 'images', 'val'),
                            stream=True,
                            verbose=False)
    for r in results:
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
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
