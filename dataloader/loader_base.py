import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from multiprocessing import Manager
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


def get_image_paths(img_dir: str) -> List[str]:
    images = []
    images.extend(sorted(
        sum([glob(os.path.join(img_dir, ext)) for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']], [])
    ))
    return images


def label_path_for(image_path: str) -> str:
    parts = image_path.split(os.sep)
    if 'images' in parts:
        parts[parts.index('images')] = 'labels'
        label_path = os.sep.join(parts)
        return '.'.join(label_path.split('.')[:-1] + ['txt'])
    return '.'.join(image_path.split('.')[:-1] + ['txt'])


class LetterBox:
    def __init__(self,
                 src_w: int,
                 src_h: int,
                 dst_w: int = 640,
                 dst_h: int = 640,
                 rect: bool = False,
                 stride: int = 32):
        self.dst_w = dst_w
        self.dst_h = dst_h
        scale = min((dst_w / src_w, dst_h / src_h))
        new_w = round(src_w * scale)
        new_h = round(src_h * scale)
        pad_w = dst_w - new_w
        pad_h = dst_h - new_h
        if rect:
            pad_w %= stride
            pad_h %= stride
        half_pad_w = pad_w / 2
        half_pad_h = pad_h / 2
        self.left = round(half_pad_w - 0.1)
        self.right = round(half_pad_w + 0.1)
        self.top = round(half_pad_h - 0.1)
        self.bottom = round(half_pad_h + 0.1)
        self.out_w = new_w + self.left + self.right
        self.out_h = new_h + self.top + self.bottom
        self.scale = scale
        self.pad = (self.left, self.top)
        self.new_shape = (new_w, new_h)

    def __call__(self,
                 image: np.ndarray) -> np.ndarray:
        if image.shape[:2][::-1] != self.new_shape:
            image = cv2.resize(image, self.new_shape, interpolation=cv2.INTER_LINEAR)
        return cv2.copyMakeBorder(image,
                                  self.top, self.bottom,
                                  self.left, self.right,
                                  cv2.BORDER_CONSTANT,
                                  value=(114, 114, 114))


class LOADER_BASE(Dataset):
    def __init__(self, args, image, label, save_path, cls2name, split_ratio=0.8):
        self.make = args.make
        self.show = args.show
        self.path = save_path
        self.images = image
        self.labels = label

        self.cls2name = cls2name
        self.name2cls = {v: k for k, v in self.cls2name.items()}

        self.letterbox = None

        n_total = len(self)
        n_train = int(n_total * split_ratio)
        n_val = n_total - n_train
        self.split_i = np.array([0] * n_train + [1] * n_val, dtype=np.int8)
        np.random.default_rng(seed=42).shuffle(self.split_i)

        if args.make == 'coco':
            manager = Manager()
            coco_dict = {
                "images": manager.list(),
                "annotations": manager.list()
            }
            self.coco_train = manager.dict(coco_dict)
            self.coco_val = manager.dict(coco_dict)

    def __len__(self):
        return len(self.images)

    def installer(self,
                  i: int,
                  image: np.ndarray,
                  label: np.ndarray):
        """

        :param i: index
        :param image: original image
        :param label: hbb
                      [[x, y, w, h, cls],
                       [x, y, w, h, cls],
                       ...]
                      obb
                      [[cls, x1, y1, x2, y2, x3, y3, x4, y4],
                       [cls, x1, y1, x2, y2, x3, y3, x4, y2],
                       ...]
        :return:
        """
        if label.dtype != 'float32':
            label = np.astype(label, np.float32)
        match self.make:
            case 'yolo':
                _, new_label_path = self.install(i, image)
                # make label
                if label.shape[1] == 5:
                    self.yolo_hbb(new_label_path,
                                  label,
                                  *image.shape[:2][::-1])
                elif label.shape[1] == 9:
                    self.yolo_obb(new_label_path,
                                  label,
                                  *image.shape[:2][::-1])
            case 'coco':
                new_image_path, _ = self.install(i, image, resize=(640, 640))
                # make label
                if label.shape[1] == 5:
                    self.coco_hbb(i, new_image_path,
                                  label,
                                  640, 640)

    def install(self,
                i: int,
                image: np.ndarray,
                path_depth: int = 3,
                resize: Tuple[int, int] = None):
        new_image_name = '.'.join(self.images[i].split('/')[-path_depth:])
        new_label_name = '.'.join(new_image_name.split('.')[:-1] + ['txt'])
        if self.split_i[i] == 0:
            tvt = 'train'
        elif self.split_i[i] == 1:
            tvt = 'val'
        else:
            tvt = 'test'
        new_image_path = os.path.join(self.path, f'images/{tvt}', new_image_name)
        new_label_path = os.path.join(self.path, f'labels/{tvt}', new_label_name)

        if resize is not None:
            if self.letterbox is None:
                self.letterbox = LetterBox(*image.shape[:2][::-1], *resize)
            image = self.letterbox(image)

        # copy image
        cv2.imwrite(new_image_path, image)

        return new_image_path, new_label_path

    @staticmethod
    def yolo_hbb(new_label_path: str,
                 labels: np.ndarray,
                 width, height
                 ):
        with open(new_label_path, 'w', encoding='utf-8') as f:
            if len(labels):
                labels[:, [0, 1]] += labels[:, [2, 3]] / 2
                if labels.max() > 1:
                    labels[:, 0:4:2] /= width
                    labels[:, 1:4:2] /= height
                for label in labels:
                    cx, cy, w, h, cls = label[:5]
                    f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(int(cls), cx, cy, w, h))

    @staticmethod
    def yolo_obb(new_label_path: str,
                 labels: np.ndarray,
                 width, height
                 ):
        with open(new_label_path, 'w', encoding='utf-8') as f:
            if len(labels):
                if labels.max() > 1:
                    labels[:, 1::2] /= width
                    labels[:, 2::2] /= height
                for label in labels:
                    cls, x1, y1, x2, y2, x3, y3, x4, y4 = label
                    f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        int(cls), x1, y1, x2, y2, x3, y3, x4, y4))

    def coco_hbb(self,
                 i: int, new_image_path: str,
                 labels: np.ndarray,
                 width: int, height: int
                 ):
        def annotate_image(coco):
            coco["images"].append({
                "id": i,
                "license": 0,
                "file_name": new_image_path.split('/')[-1],
                "height": height,
                "width": width,
                "date_captured": ""
            })

        if len(labels):
            scale = self.letterbox.M[0, 0]
            oxy = self.letterbox.M[:, 2]
            labels[:, :4] *= scale
            labels[:, :2] += oxy

            def annotate_label(box_id, label):
                x, y, w, h, cls = map(int, label)
                return {
                    "id": box_id,   # bbox id
                    "image_id": i,  # image id
                    "category_id": cls,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": [],
                    "iscrowd": 0
                }

            with ThreadPoolExecutor() as executor:
                annotations = list(executor.map(lambda args: annotate_label(*args), enumerate(labels)))
        else:
            annotations = []
        if self.split_i[i] == 0:
            annotate_image(self.coco_train)
            self.coco_train["annotations"].extend(annotations)
        elif self.split_i[i] == 1:
            annotate_image(self.coco_val)
            self.coco_val["annotations"].extend(annotations)


class LOADER(Dataset):
    def __init__(self, args):
        import yaml
        with open(os.path.join('cfg', 'datasets', args.project + '.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        self.test = args.test
        self.rect = getattr(args, 'rect', False)
        self.cls2name = {int(k): v for k, v in cfg['names'].items()}

        if self.test and 'test' not in cfg:
            raise RuntimeError(f"cfg/datasets/{args.project}.yaml must define cfg['test'] for --test")
        if args.dirs:
            img_dir = args.dirs
        else:
            img_dir = os.path.join(cfg['path'], cfg['test'])
        self.images = get_image_paths(img_dir)
        if not self.images:
            raise RuntimeError(f'No images found: {img_dir}')

        if self.test:
            self.labels = [label_path_for(path) for path in self.images]
            if not any(os.path.exists(path) for path in self.labels):
                raise RuntimeError(f"No ground-truth txt files found for cfg['test']: {img_dir}")
            missing = [path for path in self.labels if not os.path.exists(path)]
            if missing:
                raise RuntimeError(f'Missing ground-truth txt file: {missing[0]}')

    @staticmethod
    def read_yolo_label(label_path: str):
        labels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                cols = line.strip().split()
                if not cols:
                    continue
                if len(cols) not in (5, 9):
                    raise RuntimeError(f'Invalid label format: {label_path}:{line_no}')
                cls = int(float(cols[0]))
                labels.append((cls, np.asarray(cols[1:], dtype=np.float32)))
        return labels

    def __getitem__(self, i):
        path = self.images[i]
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f'Failed to read image: {path}')

        wh0 = image.shape[:2][::-1]
        letterbox = LetterBox(*wh0, 640, 640, rect=self.rect)
        image = letterbox(image)

        rescale_factor = (letterbox.scale, *letterbox.pad, letterbox.out_w, letterbox.out_h, *wh0)

        if self.test:
            return path, rescale_factor, self.read_yolo_label(self.labels[i]), image
        return path, rescale_factor, image

    def __len__(self):
        return len(self.images)
