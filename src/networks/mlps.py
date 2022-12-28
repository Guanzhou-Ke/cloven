from torch import nn

from .utils import get_norm, get_act

def build_mlp(layers, activation='relu', norm='batch', first_norm=True):
    """Build multiple linear perceptron

    Args:
        layers (list): The list of input and output dimension.
        activation (str, optional): activation function. Defaults to 'relu'.
                                    ['none', 'relu', 'softmax', 'sigmoid']
        norm (str, optional): normalization. Defaults to 'batch'.
                              `none`: not set, `batch`: denotes BatchNorm1D;
                              `layer`: denotes LayerNorm.
        first_norm (bool, optional): put the normalization layer before 
                                      non-liner activation function if True.
    """
    net = []
    for idx in range(1, len(layers)):
        net.append(
            nn.Sequential(
                nn.Linear(layers[idx-1], layers[idx]),
                get_norm(norm, num_features=layers[idx], dim=1) if first_norm else get_act(activation),
                get_act(activation) if first_norm else get_norm(norm, num_features=layers[idx], dim=1),
            )
        )
    net = nn.Sequential(*net)
    return net


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, num_layers=2):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        if self.num_layers == 3:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x
    
    
class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x