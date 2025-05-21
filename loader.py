import argparse
import json
import os

import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.loader_case import data_select

parser = argparse.ArgumentParser(description='prepare train, val dataset')
parser.add_argument('data', type=int, help='dataset option', default=0)
parser.add_argument('form', choices=['yolo', 'coco'], help='which format to convert')
parser.add_argument('--show', action='store_true', help='whether show data')
parser.add_argument('--make', action='store_true', help='whether save data')
parser.add_argument('--work', type=int, help='num of workers for multiprocessing', default=16)
args = parser.parse_args()

assert not (args.show and args.make), 'not support both show and make yet'


def main():
    # make & load dataset
    root_path, datasets = data_select(args)

    # check dataset
    if args.show:
        paused = False
        for _ in tqdm(enumerate(datasets), total=len(datasets), ncols=80):
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == 32:  # Space key
                paused = not paused

    # convert & copy dataset
    if args.make:
        print(f'{args.form} datasets saved in', root_path)

        match args.form:
            case 'yolo':
                paths = ['images/train', 'labels/train', 'images/val', 'labels/val']
            case 'coco':
                paths = ['images/train', 'images/val', 'labels']
        for path in paths:
            os.makedirs(os.path.join(root_path, path), exist_ok=True)

        dataloader = DataLoader(datasets, num_workers=args.work)
        for idx, _ in tqdm(enumerate(dataloader), total=len(datasets), ncols=80):
            pass

        if args.form == 'coco':
            def dump(filename, coco):
                coco_dict = dict(coco)
                coco_dict["images"] = list(coco_dict["images"])
                coco_dict["annotations"] = list(coco_dict["annotations"])
                with open(os.path.join(root_path, 'labels', filename), 'w', encoding='utf-8') as f:
                    json.dump(coco_dict, f, ensure_ascii=False, indent=4)

            dump('instances_train.json', datasets.coco_train)
            dump('instances_val.json', datasets.coco_val)


if __name__ == "__main__":
    main()
