from .AVSegFormer import AVSegFormer
from .AVSegFormerUFE import AVSegFormerUFE


def build_model(type, **kwargs):
    if type == 'AVSegFormer':
        return AVSegFormer(**kwargs)
    elif type == 'AVSegFormerUFE':
        return AVSegFormerUFE(**kwargs)
    else:
        raise ValueError
