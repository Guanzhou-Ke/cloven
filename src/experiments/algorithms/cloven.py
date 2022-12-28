"""
A Clustering-guided Contrastive Fusion for Multi-view Representation Learning

CLOVEN (丁香) CLustering-guided cOntrastiVE fusioN
"""
import torch
from torch import nn

from basemodel import BaseModel
from fusions import get_fusion_cls_by_name
from losses import SimCLRLoss, CClusteringLoss, DDCLoss
from networks import projection_MLP
from optimizer import AdaptedLossWeight


class CLOVEN(BaseModel):
    
    def __init__(self, args, device='cpu') -> None:
        super().__init__(args, device)
        self.can_predict = True
        self.build_encoder()
        
        # attention fusion block.
        fcls = get_fusion_cls_by_name(self.args.fusion.type)
        self.fusion_block = fcls(self.args, self.device)
        
        self.losses_num = 0
        # Clustering Module.
        if self.args.cluster_module.enable:
            self.clustering_module = ClusteringModule(self.args, self.device)
            self.class_contrastive_criterion = CClusteringLoss(self.args, self.device)
            self.geometric_criterion = DDCLoss(self.args, self.device, use_l2_flipped=True)
            self.can_predict = True
            self.losses_num += 2
            
        if self.args.train.with_gt:
            self.class_criterion = nn.CrossEntropyLoss()
            
        
        # Contrastive Module.
        if self.args.contrastive.enable:
            self.contrastive_module = projection_MLP(self.args.hidden_dim,
                                                     hidden_dim=args.contrastive.projection_hidden_dim,
                                                     out_dim=args.contrastive.projection_dim,
                                                     num_layers=args.contrastive.projection_layers)
            self.instance_contrastive_criterion = SimCLRLoss(args, device)
            self.losses_num += 1
        
        # Loss weight
        if self.args.train.enable_auto_weight:
            self.autoweight = AdaptedLossWeight(self.args, self.losses_num, self.device)
    
        self.apply(self.weights_init(self.args.backbone.init_method))
        
    def forward(self, Xs):
        hs = self.get_hs(Xs)
        z = self.fusion_block(hs)
        if self.args.cluster_module.enable:
            y, _ = self.clustering_module(z)
            return y
        else:
            return z
        
    @torch.no_grad()
    def commonZ(self, Xs):
        hs = self.get_hs(Xs)
        z = self.fusion_block(hs)
        return z

    @torch.no_grad()
    def extract_all_hidden(self, Xs):
        hs = self.get_hs(Xs)
        z = self.fusion_block(hs)
        return hs, z
    
    def __get_clustering_loss(self, z, hs):
        z_logit, z_hidden = self.clustering_module(z)
        hs_logits = []
        for h in hs:
            h_logit, _ = self.clustering_module(h)
            hs_logits.append(h_logit)
            
        class_loss = self.class_contrastive_criterion(z_logit, hs_logits)
        geom_loss = self.geometric_criterion(z_logit, z_hidden)
        
        return class_loss / self.views, geom_loss
    
    def __get_contrastive_loss(self, z, hs):
        zp = self.contrastive_module(z)
        hps = []
        for h in hs:
            hp = self.contrastive_module(h)
            hps.append(hp)
        return self.instance_contrastive_criterion(zp, hps) / self.views
    
    
    def get_loss(self, Xs, y=None, epoch=None):
        hs = self.get_hs(Xs)
        z = self.fusion_block(hs)
        
        # init vars.
        tot_loss = 0.
        contrastive_loss = 0.
        clustering_loss = 0.
        geometric_loss = 0.
        unweightloss = []
        recorder = []
        
        # 1. Calculate clustering contrastive loss.
        if self.args.cluster_module.enable:
            clustering_loss, geometric_loss = self.__get_clustering_loss(z, hs=hs)
            tot_loss = clustering_loss + geometric_loss
            unweightloss.append(clustering_loss)
            unweightloss.append(geometric_loss)
            recorder.append(('clu_loss', clustering_loss.item()))
            recorder.append(('geom_loss', geometric_loss.item()))
                  
        
        # 2. Calculate instance contrastive loss.
        if self.args.contrastive.enable:
            contrastive_loss = 0.0 * self.__get_contrastive_loss(z, hs)
            tot_loss += contrastive_loss
            recorder.append(('con_loss', contrastive_loss.item()))
            unweightloss.append(contrastive_loss)
            
        # 3. weight each loss.
        if self.args.train.enable_auto_weight:
            tot_loss = self.autoweight(unweightloss)
            for i in range(self.losses_num):
                alpha = self.autoweight.alphas[i].detach().cpu().item()
                recorder.append((f'alpha_{i}', alpha))

        return tot_loss, recorder

    @torch.no_grad()
    def predict(self, Xs):
        return self(Xs).detach().cpu().max(1)[1]
        

class ClusteringModule(nn.Module):
    
    def __init__(self, args, device='cpu') -> None:
        super().__init__()
        self.args = args
        self.in_features = self.args.hidden_dim
        self.clustering_hidden_dim = self.args.cluster_module.cluster_hidden_dim
        self.num_cluster = self.args.cluster_module.num_cluster
        self.device = device
        
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.in_features, self.clustering_hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.clustering_hidden_dim, momentum=0.1)
        )

        self.clustering_layer = nn.Sequential(
            nn.Linear(self.clustering_hidden_dim, self.num_cluster),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        h = self.hidden_layer(x)
        y = self.clustering_layer(h)
        return y, h
    