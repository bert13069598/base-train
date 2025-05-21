import importlib

from cfg import config
from dataloader import loader


def data_select(args, *dataset):
    if args.data not in loader.loader_map:
        raise ValueError('No matching dataset')

    root_path = getattr(config, loader.root_path_map[args.data])
    which = importlib.import_module(f'dataloader.loader.{loader.loader_map[args.data]}')
    which.CRAWLER(root_path, *dataset)
    datasets = which.LOADER(args, *dataset, root_path)
    return root_path, datasets
