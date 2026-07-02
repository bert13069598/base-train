import argparse
import os

from myyolo.models.yolo.model import YOLO

MODEL_CHOICES = [
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
    "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
]
parser = argparse.ArgumentParser(description='Export an PT model to a ONNX model.')
parser.add_argument('-m', '--model', type=str, help='model name for .pt',
                    choices=MODEL_CHOICES, default='yolov8s')
parser.add_argument('-o', '--obb', action='store_true', help='whether obb')
parser.add_argument('-b', '--batch', type=str, help='batch number', choices=['1', '2', '3', '4', 'd'], default='d')
parser.add_argument('-p', '--project', type=str, help='which object trained', default=None)
parser.add_argument('--rect', action='store_false', help='whether use minimal rectangle padding')
args = parser.parse_args()

if args.batch != 'd':
    args.batch = int(args.batch)

if args.obb:
    args.model += '-obb'

if args.project is not None:
    pre_model = f'runs/{args.model}/{args.project}/weights/best'  # load an custom model
    new_model = args.model
else:
    pre_model = args.model  # load an official model
    new_model = pre_model


def imgsz(src_w: int,
          src_h: int,
          dst_w: int = 640,
          dst_h: int = 640,
          rect: bool = False,
          stride: int = 32):
    scale = min((dst_w / src_w, dst_h / src_h))
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)
    pad_w = dst_w - new_w
    pad_h = dst_h - new_h
    if rect:
        pad_w %= stride
        pad_h %= stride
    return new_h + pad_h, new_w + pad_w


model = YOLO(f'{pre_model}.pt')
model.export(format='onnx',
             device='cuda:0',
             opset=17,
             imgsz=imgsz(1920, 1080, rect=args.rect),
             dynamic=args.batch == 'd',
             batch=1 if args.batch == 'd' else args.batch
             )

# rename onnx filename yoloN to yolovN
if new_model.startswith('yolo') and not new_model.startswith('yolov'):
    new_model = new_model.replace('yolo', 'yolov', 1)

for i, arg in enumerate([args.project, args.batch]):
    if arg is not None:
        if i == 1:
            new_model += f'-b'
        else:
            new_model += '-'
        new_model += str(arg)

os.rename(f'{pre_model}.onnx', f'{new_model}.onnx')
