import importlib
import os

import yaml

from dataloader import loader


def data_select(args):
    images, labels = [], []
    if args.data not in loader.loader_map:
        raise ValueError('No matching dataset')

    data = loader.data[args.data]

    with open(os.path.join('cfg', 'datasets', data + '.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    root_path = cfg['path']
    cls2name = cfg['names']

    which = importlib.import_module(f'dataloader.loader.{loader.loader_map[args.data]}')
    which.CRAWLER(root_path, images, labels)
    datasets = which.LOADER(args, images, labels, root_path, cls2name)
    return root_path, datasets
