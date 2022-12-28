"""
SwAV 
References to: https://github.com/facebookresearch/swav
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/11/15
"""
import torch
from torch import nn


class SwAVModule(nn.Module):
    
    def __init__(self, args, device) -> None:
        super().__init__()
        self.args = args
        self.input_dim = args.hidden_dim # 128
        self.projector_dim = args.contrastive.projection_dim # 2048
        self.nmb_protos = args.contrastive.nmb_protos # 256
        self.eps = args.contrastive.eps
        self.device = device
        self.ds_iters = args.contrastive.ds_iters
        self.temperature = args.contrastive.temperature
        self.projection_head = nn.Sequential(
            nn.Linear(self.input_dim , self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projector_dim, self.input_dim)
        )
        
        self.prototypes = nn.Linear(self.input_dim, self.nmb_protos, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)
    
    
    def get_loss(self, z, hs):
        
        # normalize
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
            
        z_p = self(z)
        qz = torch.exp(z_p / self.eps)
        qz = self.distributed_sinkhorn(qz, self.ds_iters, self.device)
        loss = 0
        for h in hs:
            h_p = self(h)
            ph = self.softmax(h_p / self.temperature)
            loss -= torch.mean(torch.sum(qz @ torch.log(ph), dim=1))
            
        qh = self(hs[0])
        ph = self(hs[1])
        qh = torch.exp(qh / self.eps)
        qh = self.distributed_sinkhorn(qh, self.ds_iters, self.device)
        ph = self.softmax(ph / self.temperature)
        
        loss -= torch.mean(torch.sum(qh @ torch.log(ph), dim=1))
        
        return loss
            
        
    def distributed_sinkhorn(self, Q, nmb_iters, device):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            u = torch.zeros(Q.shape[0]).to(device, non_blocking=True)
            r = torch.ones(Q.shape[0]).to(device, non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).to(device, non_blocking=True) / Q.shape[1]

            curr_sum = torch.sum(Q, dim=1)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()