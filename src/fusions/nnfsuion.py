import copy

import torch
from torch import nn

from .basic import BaseFusion


class NNFusion(BaseFusion):
    
    def __init__(self, args, device='cpu') -> None:
        super().__init__(args, device)
        act_func = self.args.fusion.activation
        views = self.args.views
        use_bn = self.args.fusion.use_bn
        mlp_layers = self.args.fusion.num_layers
        in_features = self.args.hidden_dim
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activate function must be ReLU or Tanh.')
        self.layers = [self._make_layers(in_features * views, in_features, self.act, use_bn)]
        if mlp_layers > 1:
            layers = [self._make_layers(in_features, in_features,
                                        self.act if _ < (mlp_layers - 2) else nn.Identity(),
                                        use_bn if _ < (mlp_layers - 2) else False) for _ in range(mlp_layers - 1)]
            self.layers += layers
        self.layers = nn.Sequential(*self.layers)


    def forward(self, h):
        h = torch.cat(h, dim=-1)
        z = self.layers(h)
        return z

    def _make_layers(self, in_features, out_features, act, bn=False):
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features, out_features))
        layers.append(act)
        if bn:
            layers.append(nn.BatchNorm1d(out_features))
        return nn.Sequential(*layers)
    
    
class _FusionBlock(nn.Module):
    """
    Fusion Block for Residual fusion.
    input -> (_, input_dim) -> norm -> (_, input_dim * expand) -> (_, input_dim) -> output
                   |                                                         |
                   -------------------------------+---------------------------
    """
    expand = 2
    def __init__(self, input_dim, act_func='relu', dropout=0., norm_eps=1e-5) -> None:
        super().__init__()
        latent_dim1 = input_dim * self.expand
        latent_dim2 = input_dim // self.expand
        if act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activate function must be ReLU or Tanh.')
        self.linear1 = nn.Linear(input_dim, latent_dim1, bias=False)
        self.linear2 = nn.Linear(latent_dim1, input_dim, bias=False)
        
        self.linear3 = nn.Linear(input_dim, latent_dim2, bias=False)
        self.linear4 = nn.Linear(latent_dim2, input_dim, bias=False)

        self.norm1 = nn.BatchNorm1d(input_dim, eps=norm_eps)
        self.norm2 = nn.BatchNorm1d(input_dim, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.block1(self.norm1(x))
        x = x + self.block2(self.norm2(x))
        return x
        
    def block1(self, x):
        return self.linear2(self.dropout1(self.act(self.linear1(x))))
    
    def block2(self, x):
        return self.linear4(self.dropout2(self.act(self.linear3(x))))


class ResidualFusion(BaseFusion):
    """Fusion block with residual connection.
    
       A block like to:
        input -> (_, 512) -> (_, 256) -> (_, 512) -> output
          |                                   |
          ------------------+------------------
    """
    
    def __init__(self, args, device='cpu'):
        super().__init__(args, device)
        act_func = args.fusion.activation
        views = args.views
        num_layers = args.fusion.num_layers
        in_features = args.hidden_dim
        
        self.use_bn = args.fusion.use_bn
        
        if self.use_bn:
            self.norm = nn.BatchNorm1d(in_features)
        
        
        self.map_layer = nn.Sequential(
            nn.Linear(in_features*views, in_features, bias=False),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True)
        )
        block = _FusionBlock(in_features, act_func)
        self.fusion_modules = self._get_clones(block, num_layers)
        

    def forward(self, h):
        h = torch.cat(h, dim=-1)
        # mapping view-specific feature to common feature dim.
        z = self.map_layer(h)
        # fusion.
        for mod in self.fusion_modules:
            z = mod(z)
        if self.use_bn:
            z = self.norm(z)
        return z

    def _get_clones(self, module, N):
        """
        A deep copy will take a copy of the original object and will then recursively take a copy of the inner objects. 
        The change in any of the models won’t affect the corresponding model.
        """
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    

class WeightedSortFusion(BaseFusion):
    
    def __init__(self, args, device='cpu', reduction=16) -> None:
        super().__init__(args, device)
        self.views = self.args.views
        self.channel = self.args.hidden_dim * self.views
        self.weights = None
        
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // reduction, self.channel, bias=False),
            nn.Sigmoid()
        )
        
        
        
    def __forward_weight(self, hs):
        hs = torch.cat(hs, dim=-1)
        # b, c, _ = hs.size()

        # y = self.avg_pool(hs).view(b, c)

        y = self.fc(hs)

        wt = hs * y.expand_as(hs)
        ww = torch.split(wt, self.channel // self.views, dim=1)
        wt_sum = wt.sum()
        ws = torch.tensor([w.sum() / wt_sum for w in ww]).to(self.device)
        return ws
    
    def forward(self, hs):
        
        self.weights = self.__forward_weight(hs)

        # attns = []
        
        # for idx, h in enumerate(hs):
        #     a = (self.view_qs[idx](h) @ self.view_ks[idx](h).T) @ self.view_vs[idx](h)
        #     attns.append(a)
        
        out = torch.sum(self.weights[None, None, :].detach() * torch.stack(hs, dim=-1), dim=-1)
        return out
        
        
        
def _get_clones(module, N):
    """
    A deep copy will take a copy of the original object and will then recursively take a copy of the inner objects. 
    The change in any of the models won’t affect the corresponding model.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])