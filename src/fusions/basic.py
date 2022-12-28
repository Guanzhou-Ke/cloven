from torch import nn


class BaseFusion(nn.Module):
    
    def __init__(self, args, device='cpu') -> None:
        super().__init__()
        self.args = args
        self.device = device