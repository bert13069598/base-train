import argparse
import json
import os

import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.loader_case import loader, data_select

parser = argparse.ArgumentParser(description='LOADER')
parser.add_argument('data', type=int, nargs='?', help='dataset option', default=None)
parser.add_argument('--init', type=str, help='create dataset', default='')
parser.add_argument('--show', action='store_true', help='whether show data')
parser.add_argument('--make', choices=['yolo', 'coco'], help='which format to convert')
parser.add_argument('--work', type=int, help='num of workers for multiprocessing', default=16)
parser.add_argument('--path', type=str, help='path to save the training dataset')
args = parser.parse_args()

if args.data is not None:
    print(f"{args.data} {loader.data[args.data]}")
elif args.init:
    if args.init in loader.data:
        print(f'dataset {args.init} already exists')
        exit()
    import shutil

    shutil.copy("cfg/datasets/car.yaml", f"cfg/datasets/{args.init}.yaml")
    shutil.copy("dataloader/loader/loader_car.py", f"dataloader/loader/loader_{args.init}.py")
    print(f'{args.init} created\n'
          f'1. configure dataset path: cfg/datasets/{args.init}.yaml\n'
          f'2. implement data loader: dataloader/loader/loader_{args.init}.py\n'
          f'3. verify setup by running: python loader.py {len(loader.data)} --show')
    exit()
else:
    parser.error("Either 'data' or '--init' must be provided.")

if args.show and args.make:
    parser.error("Not support both show and make yet.")


def main():
    # make & load dataset
    save_path, datasets = data_select(args)

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
        print(f'{args.make} datasets saved in', save_path)

        match args.make:
            case 'yolo':
                paths = ['images/train', 'labels/train', 'images/val', 'labels/val']
            case 'coco':
                paths = ['images/train', 'images/val', 'labels']
        for path in paths:
            os.makedirs(os.path.join(save_path, path), exist_ok=True)

        dataloader = DataLoader(datasets, num_workers=args.work)
        for idx, _ in tqdm(enumerate(dataloader), total=len(datasets), ncols=80): pass

        if args.make == 'coco':
            categories = [{"id": k, "name": v} for k, v in datasets.cls2name.items()]

            def dump(filename, coco):
                coco_dict = dict(coco)
                coco_dict["images"] = list(coco_dict["images"])
                coco_dict["annotations"] = list(coco_dict["annotations"])
                coco_dict["categories"] = categories
                with open(os.path.join(save_path, 'labels', filename), 'w', encoding='utf-8') as f:
                    json.dump(coco_dict, f, ensure_ascii=False, indent=4)

            dump('instances_train.json', datasets.coco_train)
            dump('instances_val.json', datasets.coco_val)


if __name__ == "__main__":
    main()
