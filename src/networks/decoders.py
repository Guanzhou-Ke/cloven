from torch import nn
from . import build_mlp
from .utils import get_norm, get_act

# -------------------------------------
# |
# |   Build decoder for reconstruction. 
# |
# -------------------------------------
class AdaptedDecoder(nn.Module):
    """Build decoder for reconstruction. """
    
    def __init__(self, 
                 type, 
                 layers, 
                 first_norm=False,
                 activation='relu',
                 output_shape=None,
                 kernals=None, 
                 strides=None, 
                 out_paddings=None) -> None:
        """
        Build decoder for reconstruction. 
        
        Args:
            type (str): the network type of decoder, must in ['cnn', 'mlp'].
            layers (int): how much layers use to reconstruction.
            output_shape (tuple): the output shape of image.
            kernals (tuple, optional): For ConvTranspose2d.
            strides (tuple, optional): For ConvTranspose2d.
            out_paddings (tuple, optional): For ConvTranspose2d.
            
        Return:
            
        """
        super().__init__()
        self.type = type
        self.layers = layers
        self.first_norm = first_norm
        self.activation = activation
        # For CNN
        self.output_shape = output_shape
        self.kernals = kernals
        self.strides = strides
        self.out_paddings = out_paddings
        
        self.net = self.__build_decoder()
        
    def forward(self, x):
        if self.type == 'cnn':
            bs, f = x.shape
            x = x.view(bs, f, 1, 1)
        recon_x = self.net(x)
        return recon_x
    
    def __build_decoder(self):
        if self.type == 'mlp':
            net = build_mlp(self.layers, activation=self.activation, first_norm=self.first_norm)
        elif self.type == 'cnn':
            unified_kernal = isinstance(self.kernals, int)
            unified_stride = isinstance(self.strides, int)
            unified_output_padding = isinstance(self.out_paddings, int)
            net = []
            for idx in range(1, len(self.layers)):
                net.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(self.layers[idx-1], self.layers[idx], 
                                           kernel_size=self.kernals if unified_kernal else self.kernals[idx-1],
                                           stride=self.strides if unified_stride else self.strides[idx-1],
                                           output_padding=self.out_paddings if unified_output_padding else self.out_paddings[idx-1]),
                        get_norm('batch', num_features=self.layers[idx], dim=2) if self.first_norm else get_act(self.activation),
                        get_act(self.activation) if self.first_norm else get_norm('batch', num_features=self.layers[idx], dim=2),
                    )
                )
            net = nn.Sequential(*net,
                                nn.AdaptiveAvgPool2d(self.output_shape))
        else:
            raise ValueError("Decoder type must be 'mlp' or 'cnn'.")
        return net