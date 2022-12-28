"""
Backbones collections.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/11/15
"""

from .mlps import projection_MLP, build_mlp, prediction_MLP
from .swav import SwAVModule
from .cnns import build_cnn2d, NormalCNN
from .resnet import *
from .decoders import AdaptedDecoder
from .utils import get_norm, get_act
from .position_embedding import SinusoidalPositionalEmbedding
from .transformer import TransformerEncoder, TransformerEncoderLayer


__all__ = ['projection_MLP', 'SwAVModule', 'build_mlp', 'prediction_MLP', 
           'build_cnn2d', 'NormalCNN', 'build_off_the_shelf_cnn', 'AdaptedDecoder',
           'get_norm', 'get_act', 'SinusoidalPositionalEmbedding', 'TransformerEncoder',
           'TransformerEncoderLayer']


def build_off_the_shelf_cnn(name='resnet12', 
                            fc_identity=True, 
                            num_classes=1000, 
                            channels=3):
    """
    Build off-the-shelf CNN networks, likes ResNet, VGG...
    """
    builders = {
        'resnet12': ResNet12,
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'normal': NormalCNN
    }
    func = builders.get(name, None)
    if func is None:
        raise ValueError(f"{name} is not a available network name.")
    backbone = func(num_classes, channels, fc_identity)
    return backbone


