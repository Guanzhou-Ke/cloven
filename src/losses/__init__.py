"""
Losses collections.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/11/15
"""
from .baseloss import BaseLoss
from .simclrloss import SimCLRLoss
from .ddcloss import DDCLoss
from .cclusteringloss import CClusteringLoss


__all__ = ['SimCLRLoss', 
           'DDCLoss', 'CClusteringLoss', 'BaseLoss']
    
    
if __name__ == '__main__':
    pass
    