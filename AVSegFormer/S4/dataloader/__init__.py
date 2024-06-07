from .s4_dataset import S4Dataset
from .s4_dataset_ufe import S4DatasetUFE
from mmengine.config import Config


def build_dataset(type, split, label=False, **kwargs):
    if type == 'S4Dataset':
        return S4Dataset(split=split, cfg=Config(kwargs))
    elif type == 'S4DatasetUFE':
        return S4DatasetUFE(label=label, split=split, cfg=Config(kwargs))
    else:
        raise ValueError


__all__ = ['build_dataset', 'get_v2_pallete']
