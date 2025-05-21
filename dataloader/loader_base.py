import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from torch.utils.data import Dataset
from typing import Tuple


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
    def __init__(self, args, image, label, make_path, split_ratio=0.8):
        self.make = args.make
        self.show = args.show
        self.form = args.form
        self.path = make_path
        self.images = image
        self.labels = label

        self.letterbox = None

        n_total = len(self)
        n_train = int(n_total * split_ratio)
        n_val = n_total - n_train
        self.split_i = np.array([0] * n_train + [1] * n_val, dtype=np.int8)
        np.random.default_rng(seed=42).shuffle(self.split_i)

        if args.form == 'coco':
            manager = Manager()
            self.coco_train = manager.dict({
                "categories": [],
                "images": manager.list(),
                "annotations": manager.list()
            })
            self.coco_val = manager.dict({
                "categories": [],
                "images": manager.list(),
                "annotations": manager.list()
            })

    def __len__(self):
        return len(self.images)

    def installer(self,
                  i: int,
                  image: np.ndarray,
                  label: np.ndarray):
        """

        :param i: index
        :param image: original image
        :param label: [[x, y, w, h, cls],
                       [x, y, w, h, cls],
                       ...]
        :return:
        """
        match self.form:
            case 'yolo':
                _, new_label_path = self.install(i, image)
                # make label
                self.yolo_hbb(new_label_path,
                              label,
                              *image.shape[:2][::-1])
            case 'coco':
                new_image_path, _ = self.install(i, image, resize=(640, 640))
                # make label
                self.coco_hbb(i, new_image_path,
                              label,
                              640, 640)

    def install(self,
                i: int,
                image: np.ndarray,
                path_depth: int = 3,
                resize: Tuple[int, int] = None):
        new_image_name = '.'.join(self.images[i].split('/')[-path_depth:])
        if isinstance(self.labels[i], str):
            new_label_name = '.'.join(self.labels[i].split('/')[-path_depth:])
            new_label_name = new_label_name.replace('json', 'txt')
        else:
            new_label_name = new_image_name.replace('jpg', 'txt')
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
            for label in labels:
                if len(label):
                    x, y, w, h, cls = label[:5]
                    cx, cy, w, h = (x + w / 2) / width, (y + h / 2) / height, w / width, h / height
                    f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(cls, cx, cy, w, h))

    @staticmethod
    def yolo_obb(new_label_path: str,
                 labels: np.ndarray,
                 width, height
                 ):
        with open(new_label_path, 'w', encoding='utf-8') as f:
            if labels.shape[0]:
                for label in labels:
                    cls = label[:, 0]
                    xyxyxyxys = label[:, 1:]

                    xyxyxyxys[:, 0::2] /= width
                    xyxyxyxys[:, 1::2] /= height
                    for c, xyxyxyxy in zip(cls, xyxyxyxys):
                        f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            int(c),
                            xyxyxyxy[0], xyxyxyxy[1], xyxyxyxy[2], xyxyxyxy[3],
                            xyxyxyxy[4], xyxyxyxy[5], xyxyxyxy[6], xyxyxyxy[7]))

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
                    "category_id": cls + 1,
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
