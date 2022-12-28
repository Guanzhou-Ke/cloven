"""
DDC Loss
References to Michael Kampffmeyer et al. "Deep divergence-based approach to clustering"
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/11/15
"""

import torch
from torch.nn import functional as F

from . import BaseLoss


class DDCLoss(BaseLoss):

    def __init__(self, args, 
                 device='cpu', 
                 epsilon=1e-9, 
                 rel_sigma=0.15,
                 use_l2_flipped=False):
        """
        :param epsilon:
        :param rel_sigma: Gaussian kernel bandwidth
        """
        super(DDCLoss, self).__init__(args, device)
        self.epsilon = epsilon
        self.rel_sigma = rel_sigma
        # EAMC use l2_flipped
        self.use_l2_flipped = use_l2_flipped 
        self.num_cluster = self.args.cluster_module.num_cluster

    def __call__(self, logist, hidden):
        hidden_kernel = self._calc_hidden_kernel(hidden)
        l1_loss = self._l1_loss(logist, hidden_kernel, self.num_cluster)
        l2_loss = self._l2_flipped(logist) if self.use_l2_flipped else self._l2_loss(logist)
        l3_loss = self._l3_loss(logist, hidden_kernel, self.num_cluster)
        return l1_loss + l2_loss + l3_loss

    def _l1_loss(self, logist, hidden_kernel, num_cluster):
        return self._d_cs(logist, hidden_kernel, num_cluster)

    def _l2_loss(self, logist):
        n = logist.size(0)
        return 2 / (n * (n - 1)) * self._triu(logist @ torch.t(logist))
    
    def _l2_flipped(self, logist):
        return 2 / (self.num_cluster * (self.num_cluster - 1)) * self._triu(torch.t(logist) @ logist)

    def _l3_loss(self, logist, hidden_kernel, num_cluster):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(num_cluster, device=self.device)
        m = torch.exp(-self._cdist(logist, self.eye))
        return self._d_cs(m, hidden_kernel, num_cluster)

    def _triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _calc_hidden_kernel(self, x):
        return self._kernel_from_distance_matrix(self._cdist(x, x), self.epsilon)

    def _d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.
        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom, eps=self.epsilon)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=self.epsilon ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self._triu(nom / torch.sqrt(dnom_squared))
        return d

    def _atleast_epsilon(self, X, eps):
        """
        Ensure that all elements are >= `eps`.
        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def _cdist(self, X, Y):
        """
        Pairwise distance between rows of X and rows of Y.
        :param X: First input matrix
        :type X: torch.Tensor
        :param Y: Second input matrix
        :type Y: torch.Tensor
        :return: Matrix containing pairwise distances between rows of X and rows of Y
        :rtype: torch.Tensor
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    def _kernel_from_distance_matrix(self, dist, min_sigma):
        """
        Compute a Gaussian kernel matrix from a distance matrix.
        :param dist: Disatance matrix
        :type dist: torch.Tensor
        :param min_sigma: Minimum value for sigma. For numerical stability.
        :type min_sigma: float
        :return: Kernel matrix
        :rtype: torch.Tensor
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = self.rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k
    

if __name__ == '__main__':
    pass