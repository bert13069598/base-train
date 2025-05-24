import argparse
import os.path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

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

model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')

if args.auto:
    with open(os.path.join('cfg', 'datasets', args.project + '.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    results = model.predict(source=os.path.join(cfg['path'], 'images', 'test', '*.png'),
                            name=args.project,
                            project=f'runs/{args.model}',
                            stream=True,
                            batch=args.work,
                            save_txt=True,
                            verbose=False)
    for r in tqdm(enumerate(results), total=1815, ncols=80): pass

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
