from torch import nn

from .utils import get_norm, get_act


def build_cnn2d(layers, 
                kernals, 
                activation='relu', 
                norm='batch', 
                first_norm=True,
                max_pooling=False):
    """
    Build CNN

    Args:
        layers (list): The list of input and output dimension.
        kernals (int or list): using unified kernal size for cnn if kernals is `int`, 
                               else must correspond to the layers, the size equal to `len(layers) - 2`.
        activation (str, optional): activation function. Defaults to 'relu'.
                                    ['none', 'relu', 'softmax', 'sigmoid']
        norm (str, optional): normalization. Defaults to 'batch'.
                              `none`: not set, `batch`: denotes BatchNorm1D;
                              `layer`: denotes LayerNorm.
        first_norm (bool, optional): put the normalization layer before 
                                      non-liner activation function if True.
        max_pooling (bool, optional): add max pooling to last network block.
    """
    unified_kernal = isinstance(kernals, int)
    net = []
    for idx in range(1, len(layers)):
        net.append(
            nn.Sequential(
                nn.Conv2d(layers[idx-1], layers[idx], kernel_size=kernals if unified_kernal else kernals[idx-1]),
                get_norm(norm, num_features=layers[idx], dim=2) if first_norm else get_act(activation),
                get_act(activation) if first_norm else get_norm(norm, num_features=layers[idx], dim=2),
            )
        )
    net = nn.Sequential(
        *net,
        nn.AdaptiveAvgPool2d(1) if max_pooling else nn.Identity()
    )
    return net


class NormalCNN(nn.Module):
    def __init__(self, num_classese=10, channels=3, fc_identity=True):
        super(NormalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 5, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        if fc_identity:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(288, num_classese)

    def forward(self, x):
        x = self.features(x)
        y = self.fc(x)
        return y   