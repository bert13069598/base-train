import argparse
import os.path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='TEST')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov8s')
parser.add_argument('-o', '--obb', action='store_true', help='whether obb')
parser.add_argument('-p', '--project', type=str, help='which object trained', default=None)
parser.add_argument('--show', action='store_true', help='whether show')
parser.add_argument('--auto', action='store_true', help='whether auto labeling')
args = parser.parse_args()

with open(os.path.join('cfg', 'datasets', args.project + '.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)
root_path = cfg['path']
cls2name = cfg['names']

if args.obb:
    args.model += '-obb'
model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')

results = model(source=os.path.join(root_path, 'images', 'val'),
                stream=True,
                verbose=False)
for r in results:
    if r.obb is not None:
        obb = r.obb
        if args.show:
            for cls, box in zip(obb.cls, obb.xyxyxyxy.cpu()):
                cv2.polylines(r.orig_img, [np.asarray(box, dtype=int)], True, (0, 255, 0), 2)
                x1, y1 = box[0].int()
                cv2.putText(r.orig_img, cls2name[cls.item()], (x1.item(), y1.item() - 5), 0, 1, (0, 255, 0), 2, 16)
        if args.auto:
            path = '.'.join(r.path.split('.')[:-1] + ['txt'])
            with open(path, 'w', encoding='utf-8') as f:
                for cls, box in zip(obb.cls, obb.xyxyxyxyn):
                    x1, y1, x2, y2, x3, y3, x4, y4 = box.flatten().cpu().numpy()
                    f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        int(cls.item()), x1, y1, x2, y2, x3, y3, x4, y4))
    else:
        hbb = r.boxes
        if args.show:
            for cls, box in zip(hbb.cls, hbb.xyxy):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(r.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(r.orig_img, cls2name[cls.item()], (x1, y1 - 5), 0, 1, (0, 255, 0), 2, 16)
        if args.auto:
            path = '.'.join(r.path.split('.')[:-1] + ['txt'])
            with open(path, 'w', encoding='utf-8') as f:
                for cls, box in zip(hbb.cls, hbb.xywhn):
                    cx, cy, w, h = box.cpu().numpy()
                    f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(int(cls), cx, cy, w, h))
    if args.show:
        cv2.imshow('sample', r.orig_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
