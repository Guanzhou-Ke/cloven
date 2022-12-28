import sys
# 3.8 supported
# from math import prod

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cluster import KMeans
from munkres import Munkres


def get_masked(batch_size, shapes, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.
        Args:
          batch_size: the ba
          shapes: the shape of data.
          missing_rate: missing ratio. [0, 1]
        Returns: 
          mask: torch.ByteTensor
    """
    masks = []
    for shape in shapes:
        mask = np.r_[[np.random.choice([0, 1], size=shape, p=[1-missing_rate, missing_rate]) for _ in range(batch_size)]]
        masks.append(torch.BoolTensor(mask))
    return masks


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    print('Global seed:', seed)

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(f'Total number of parameters: {num_params}, size: {num_params/1e6*32/8:.2f} M')


def clustering_by_representation(X_rep, y):
    """Get scores of clustering by representation"""
    n_clusters = np.size(np.unique(y))

    kmeans_assignments, _ = get_cluster_sols(X_rep, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10})
    if np.min(y) == 1:
        y = y - 1
    return clustering_metric(y, kmeans_assignments)
    


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    """Get classification metric"""
   
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return accuracy, precision, f_score


def clustering_metric(y_true, y_pred, decimals=4):
    """Get clustering metric"""
    n_clusters = np.size(np.unique(y_true))
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)
    
    class_acc, p, fscore = classification_metric(y_true, y_pred_ajusted)
    
    # ACC
    acc = clustering_accuracy(y_true, y_pred)
    acc = np.round(acc, decimals)
    
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return acc, nmi, ari, class_acc, p, fscore

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """Computes the predicted labels, where label assignments now
        correspond to the actual labels in y_true (as estimated by Munkres)

        Args:
            cluster_assignments: array of labels, outputted by kmeans
            y_true:              true labels
            n_clusters:          number of clusters in the dataset

        Returns:
            a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Using either a newly instantiated ClusterClass or a provided cluster_obj, generates
        cluster assignments based on input data.

        Args:
            x: the points with which to perform clustering
            cluster_obj: a pre-fitted instance of a clustering class
            ClusterClass: a reference to the sklearn clustering class, necessary
              if instantiating a new clustering class
            n_clusters: number of clusters in the dataset, necessary
                        if instantiating new clustering class
            init_args: any initialization arguments passed to ClusterClass

        Returns:
            a tuple containing the label assignments and the clustering object
    """
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def classify_via_svm(train_X, train_Y, test_X, test_Y, **kwargs):
    """
    A simple classifier for representation learning. (SVM)
    ---
    Args:
      train_X: (N, D) training matrix. 
      train_Y: (N, 1) training labels.
      test_X: (M, D) test matrix.
      test_Y: (M, 1) test labels.
      
    Return:
      A result values (@acc, @percision, @fscore)
    """
    # We suggest to use the default setting.
    clf = SVC(**kwargs)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test_X)
    acc, p, fscore = classification_metric(test_Y, preds)
    return acc, p, fscore


def classify_via_vote(train_X, train_Y, test_X, test_Y, n=1):
    """
    References to: https://github.com/XLearning-SCU/2022-TPAMI-DCP/blob/26a67a693ab7392a8c9e002b96f90137ea7fd196/utils/classify.py#L9
    Sometimes the prediction accuracy will be higher in this way.
    ---
    Args: 
      train_X: train set's latent space data
      train_Y: label of train set
      test_X: test set's latent space data
      test_Y: label of test set
      n: Similar to K in k-nearest neighbors algorithm
    
    Return: 
      A result values (@acc, @percision, @fscore)
    """
    F_h_h = np.dot(test_X, np.transpose(train_X))
    gt_list = []
    train_Y = train_Y.reshape(len(train_Y), 1)
    for _ in range(n):
        F_h_h_argmax = np.argmax(F_h_h, axis=1)
        F_h_h_onehot = convert_to_one_hot(F_h_h_argmax, len(train_Y))
        F_h_h = F_h_h - np.multiply(F_h_h, F_h_h_onehot)
        gt_list.append(np.dot(F_h_h_onehot, train_Y))
    gt_ = np.array(gt_list).transpose(2, 1, 0)[0].astype(np.int64)
    count_list = []
    count_list.append([np.argmax(np.bincount(gt_[i])) for i in range(test_X.shape[0])])
    gt_pre = np.array(count_list).transpose()
    
    acc, p, fscore = classification_metric(test_Y, gt_pre)
    return acc, p, fscore
    
    
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


if __name__ == '__main__':
    pass
    # from collections import Counter
    # pred1 = torch.randint(0, 10, (100, ))
    # counter1 = Counter(pred1.tolist())
    # most1 = counter1.most_common(2)
    # m1, m2 = most1[0][0], most1[1][0]
    # print(most1, m1, m2)
    # pred2 = []
    # for e in pred1:
    #     if e == m1:
    #         pred2.append(m2)
    #     elif e == m2:
    #         pred2.append(m1)
    #     else:
    #         pred2.append(e)
    # most2 = Counter(pred2).most_common(2)
    # print(most2)
    # pred2 = torch.tensor(pred2)
    
    # kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    # dist_p1 = kl_loss(pred1, pred1)
    # print('p1, p1', dist_p1)
    
    # dist_p2 = kl_loss(pred2, pred2)
    # print('p2, p2', dist_p2)
    
    # dist = kl_loss(pred1, pred2)
    # print('p1, p2', dist)
    
    
    