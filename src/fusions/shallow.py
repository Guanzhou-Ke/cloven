"""
Shallow Fusions
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/12/21
"""
import torch
from torch import nn

from .basic import BaseFusion


class ConcatenateFusion(BaseFusion):
    """
    Concatenate fusion method.
    """
        
    def __init__(self, args, device='cpu') -> None:
        super().__init__(args, device)

    def forward(self, hs):
        z = torch.concat(hs, dim=-1)
        return z


class WeightsumFusion(BaseFusion):
    """
    Weight-sum fusion method.
    """
    def __init__(self, args, device='cpu') -> None:
        super().__init__(args, device)
        self.views = self.args.views
        self.weights = nn.Parameter(torch.rand(self.views, 1))

    def forward(self, hs):
        hs = [h.unsqueeze(-1) for h in hs]
        hs = torch.concat(hs, dim=-1)
        z = hs @ self.weights
        return z.squeeze()