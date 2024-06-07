from .ms3_dataset import MS3Dataset
from .ms3_dataset_ufe import MS3DatasetUFE
from mmengine.config import Config


def build_dataset(type, split, label=False, **kwargs):
    if type == 'MS3Dataset':
        return MS3Dataset(split=split, cfg=Config(kwargs))
    elif type == 'MS3DatasetUFE':
        return MS3DatasetUFE(split=split, label=label, cfg=Config(kwargs))
    else:
        raise ValueError


__all__ = ['build_dataset', 'get_v2_pallete']
