"""
Contrastive Clustering Module. [Modified]
References to: https://github.com/Yunfan-Li/Contrastive-Clustering/blob/main/modules/contrastive_loss.py
----

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/11/15
"""
import math

import torch

from . import BaseLoss


class CClusteringLoss(BaseLoss):
    
    def __init__(self, args, device='cpu') -> None:
        super().__init__(args, device)
        self.num_cluster = self.args.cluster_module.num_cluster
        self.temperature = self.args.contrastive.temperature
        self.mask = self.mask_correlated_clusters(self.num_cluster)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)
    
    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask
    
    def get_loss(self, z_logit, hs_logit):
        # calc contrastive loss
        con_loss = 0.
        ne_loss = 0.
        for hlog in hs_logit:
            ne_loss += self.__H_loss(z_logit, hlog)
            con_loss += self.__contrastive_clustering_loss(z_logit, hlog)
        
        total_loss = con_loss + ne_loss
        return total_loss
    
    def __H_loss(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j
        return ne_loss
    
    
    def __contrastive_clustering_loss(self, c_i, c_j):
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.num_cluster
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.num_cluster)
        sim_j_i = torch.diag(sim, -self.num_cluster)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss