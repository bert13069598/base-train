import argparse
import os.path

import cv2
from ultralytics import YOLO

from cfg.config import root_balloon

parser = argparse.ArgumentParser(description='YOLO TRAIN')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov8s')
parser.add_argument('-o', '--obb', action='store_true', help='whether obb')
parser.add_argument('-p', '--project', type=str, help='which object trained', default=None)
parser.add_argument('-d', '--dataset', type=str, help='which dataset eval', default=None)
args = parser.parse_args()

if args.obb:
    args.model += '-obb'
model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')

results = model(source=os.path.join(root_balloon, f'{args.dataset}'), stream=True, verbose=False)
for r in results:
    boxes = r.boxes
    if boxes.xyxy.shape[0]:
        for box, cls in zip(boxes.xyxy, boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(r.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('sample', r.orig_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
