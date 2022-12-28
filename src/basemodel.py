"""
Model.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import math
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import init

from networks import (build_mlp, 
                      build_cnn2d, 
                      build_off_the_shelf_cnn,
                      AdaptedDecoder)


class BaseModel(nn.Module, ABC):
    def __init__(self, args, device='cpu') -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.views = self.args.views
        # For the method which can predict label, like conan.
        self.can_predict = False
    
    
    def build_decoder(self):
        if self.args.backbone.type == 'mlp':
            if self.args.backbone.shared:
                self.decoder = AdaptedDecoder('mlp', self.args.backbone.decoders[0], first_norm=self.args.backbone.first_norm)
            else:
                for idx, layers in enumerate(self.args.backbone.decoders):
                    bb = AdaptedDecoder('mlp', layers, first_norm=self.args.backbone.first_norm)
                    self.__setattr__(f'decoder{idx}', bb)
        else:
            for idx, layers in enumerate(self.args.backbone.decoders):
                bb = AdaptedDecoder('cnn',
                                    layers,
                                    first_norm=self.args.backbone.first_norm,
                                    output_shape=(self.args.backbone.input_shapes[idx][1], self.args.backbone.input_shapes[idx][2]),
                                    kernals=self.args.backbone.decoder_kernals[idx],
                                    strides=self.args.backbone.decoder_strides[idx],
                                    out_paddings=self.args.backbone.decoder_output_paddings[idx])
                self.__setattr__(f'decoder{idx}', bb)
    
    
    def build_encoder(self):
        if self.args.backbone.type == 'mlp':
            if self.args.backbone.shared:
                # all views share an encoder.
                self.encoder = build_mlp(self.args.backbone.encoders[0], first_norm=self.args.backbone.first_norm)
            else:
                for idx, layers in enumerate(self.args.backbone.encoders):
                    bb = build_mlp(layers, first_norm=self.args.backbone.first_norm)
                    self.__setattr__(f'encoder{idx}', bb)
        else:
            # For CNN
            if isinstance(self.args.backbone.encoders[0], list):
                if self.args.backbone.shared:
                    self.encoder = build_cnn2d(self.args.backbone.encoders[0],
                                            self.args.backbone.kernals[0],
                                            first_norm=self.args.backbone.first_norm,
                                            max_pooling=self.args.backbone.max_pooling)
                else:
                    for idx, layers in enumerate(self.args.backbone.encoders):
                        bb = build_cnn2d(layers,
                                        self.args.backbone.kernals[idx],
                                        first_norm=self.args.backbone.first_norm,
                                        max_pooling=self.args.backbone.max_pooling)
                        self.__setattr__(f'encoder{idx}', bb)
            else:
                if self.args.backbone.shared:
                    self.encoder = build_off_the_shelf_cnn(self.args.backbone.encoders[0],
                                                           channels=self.args.backbone.channels,
                                                           fc_identity=self.args.backbone.fc_identity)
                else:
                    for idx, name in enumerate(self.args.backbone.encoders):
                        bb = build_off_the_shelf_cnn(name,
                                                     channels=self.args.backbone.channels,
                                                     fc_identity=self.args.backbone.fc_identity)
                        self.__setattr__(f'encoder{idx}', bb)
                        
    def enc_dec(self, Xs):
        """
        encoder + decoder.
        """
        hs = self.get_hs(Xs)
        Xs_recon = self.recon_xs(hs)
        return Xs_recon
    
    def get_hs(self, Xs):
        # Preprocess, image to vector
        if self.args.backbone.type == 'mlp' and len(Xs[0].shape) > 2:
            bs = Xs[0].shape[0]
            Xs = [x.view(bs, -1) for x in Xs]
        
        if self.args.backbone.shared:
            hs = [self.encoder(x) for x in Xs]
        else:
            hs = [bb(x) for bb, x in zip([self.__getattr__(f"encoder{idx}") for idx in range(self.views)], Xs)]
            
        # backprocess feature map to vector.
        if self.args.backbone.type == 'cnn':
            bs = Xs[0].shape[0]
            hs = [h.view(bs, -1) for h in hs]
        return hs
    
    def recon_xs(self, hs):
        """
        Reconstruct original Xs via hs.
        """
        input_shapes = self.args.backbone.input_shapes
        bs = hs[0].size(0)  
        if self.args.backbone.shared:
            if input_shapes:
                recon_Xs = [self.decoder(h).view(bs, *input_shapes[idx]) for idx, h in enumerate(hs)]
            else:
                recon_Xs = [self.decoder(h) for h in hs]
        else:
            if input_shapes:
                recon_Xs = [bb(h).view(bs, *input_shapes[idx]) for idx, (bb, h) in enumerate(zip([self.__getattr__(f"decoder{idx}") for idx in range(self.views)], hs))]
            else:
                recon_Xs = [bb(h) for bb, h in zip([self.__getattr__(f"decoder{idx}") for idx in range(self.views)], hs)]
        return recon_Xs
    
    @abstractmethod
    def get_loss(self, Xs, y=None, epoch=None):
        """
        Return total loss and loss partial
        Example:
        return total_loss, ((loss name, loss value), ....)
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def commonZ(self, Xs):
        pass
    
    @torch.no_grad()
    def extract_all_hidden(self, Xs):
        pass

    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0.0)

        return init_fun

    
if __name__ == '__main__':
    pass