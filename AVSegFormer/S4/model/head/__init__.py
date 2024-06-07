from .AVSegHead import AVSegHead
from .AVSegHeadUFE import AVSegHeadUFE

def build_head(type, **kwargs):
    if type == 'AVSegHead':
        return AVSegHead(**kwargs)
    elif type == 'AVSegHeadUFE':
        return AVSegHeadUFE(**kwargs)
    else:
        raise ValueError


__all__ = ['build_head']
