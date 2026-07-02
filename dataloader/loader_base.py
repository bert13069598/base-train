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
                 dst_h: int = 640):
        self.dst_w = dst_w
        self.dst_h = dst_h
        scale = min((dst_w / src_w, dst_h / src_h))
        ox = (dst_w - scale * src_w) / 2
        oy = (dst_h - scale * src_h) / 2
        self.M = np.array([
            [scale, 0, ox],
            [0, scale, oy]
        ], dtype=np.float32)

    def __call__(self,
                 image: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(image, self.M,
                              (self.dst_w, self.dst_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(114, 114, 114))


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
        letterbox = LetterBox(*wh0, 640, 640)
        image = letterbox(image)

        w0, h0 = wh0
        w_h = w0 > h0
        r = max(w0, h0) / min(w0, h0)
        pad = abs(w0 - h0) / 2 / min(w0, h0)
        rescale_factor = w_h, r, pad

        if self.test:
            return path, rescale_factor, self.read_yolo_label(self.labels[i]), image
        return path, rescale_factor, image

    def __len__(self):
        return len(self.images)
