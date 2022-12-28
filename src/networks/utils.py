from torch import nn


def get_act(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax':
        return nn.Softmax(dim=-1)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'leaky-relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Activation function name error: {name}")
    

def get_norm(name, num_features, dim=1):
    if name == None:
        return nn.Identity()
    elif name == 'batch':
        return nn.BatchNorm1d(num_features) if dim == 1 else nn.BatchNorm2d(num_features)
    elif name == 'layer':
        return nn.LayerNorm(num_features)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Normalization name erro: {name}")