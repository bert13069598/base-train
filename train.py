from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='TRAIN')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov8s')
parser.add_argument('-e', '--epoch', type=int, help='model name for .pt', default=20)
parser.add_argument('--imgsz', type=int, help='model name for .pt', default=640)
parser.add_argument('-o', '--obb', action='store_true', help='whether obb')
parser.add_argument('-p', '--project', type=str, help='which object trained', default=None)
args = parser.parse_args()

# Load a model
if args.obb:
    args.model += '-obb'
model = YOLO(f'{args.model}.pt')

# Train the model
results = model.train(
    data=f'cfg/datasets/{args.project}.yaml',
    epochs=args.epoch,
    imgsz=args.imgsz,
    name=args.project,
    project=f'runs/{args.model}',
    device=0
)
