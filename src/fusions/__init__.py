from .shallow import ConcatenateFusion, WeightsumFusion
from .nnfsuion import NNFusion, ResidualFusion

__all__ = ['get_fusion_cls_by_name', ConcatenateFusion, WeightsumFusion, 
           NNFusion, ResidualFusion]


def get_fusion_cls_by_name(name):
    __fun_dict = {
        'concat': ConcatenateFusion,
        'weight-sum': WeightsumFusion,
        'nn': NNFusion,
        'resfus': ResidualFusion,
    }
    fcls = __fun_dict.get(name, None)
    if fcls is None:
        raise ValueError(f"The `{name}` is not a legal fusion module.")
    else:
        return fcls