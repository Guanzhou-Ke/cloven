"""
SimCLR Loss
References to Ting Chen's SimCLR.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/11/15
"""
import torch

from . import BaseLoss


class SimCLRLoss(BaseLoss):
    """
    Multi-view setting SimCLR
    """
    
    def __init__(self, args, device='cpu') -> None:
        super().__init__(args, device)
        self.batch_size = self.args.train.batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)
        self.temp = self.args.contrastive.temperature
        self.symmetry = self.args.contrastive.symmetry
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        
    def get_loss(self, zp, hps):
        """
        Args:
           zp: vector, view-common representation $z$ passed by prejector.
           hps: list, lisf of view-specific representation $h$ passed by prejector.
        Return:
           loss value.
        """
        loss = 0.
        if zp is not None:
            for hp in hps:
                loss += self._loss_func(zp, hp)
                
            if self.symmetry:
                l = len(hps)
                for i in range(l):
                    for j in range(i+1, l):
                        h1_p = hps[i]
                        h2_p = hps[j]
                        loss += self._loss_func(h1_p, h2_p)
        else:
            l = len(hps)
            for i in range(l):
                for j in range(i+1, l):
                    h1_p = hps[i]
                    h2_p = hps[j]
                    loss += self._loss_func(h1_p, h2_p)
            
        
        return loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    
    def _loss_func(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temp
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss