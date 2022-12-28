"""
For evaluate single view method.
---

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/11/30
"""
import argparse
from time import time

import numpy as np
from sklearn.decomposition import PCA

from datatool import load_dataset
from config import get_cfg_defaults
from utils import clustering_by_representation, classify_via_svm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', '-d', type=str, help='dataset name')
    parser.add_argument('--dim', '-n', type=int, help='the dimensionality of pca.')
    args = parser.parse_args()
    return args


def single_view_clustering(X, y):
    acc, nmi, ari, class_acc, p, fscore = clustering_by_representation(X, y)
    return acc, nmi, ari


def single_view_classification(train_X, train_Y, test_X, test_Y):
    acc, p, fscore = classify_via_svm(train_X, train_Y, test_X, test_Y)
    return acc, p, fscore


def main():
    args = parse_args()
    dim = args.dim
    dataname = args.data_name
    config = get_cfg_defaults()
    config.dataset.name = dataname
    config.dataset.split = True
    train_dataset, test_dataset = load_dataset(config)
    train_Y = train_dataset.targets.numpy()
    test_Y = test_dataset.targets.numpy()
    n_views = len(train_dataset.data)
    train_conatenatings = []
    test_conatenatings = []
    for i in range(n_views):
        train_X = train_dataset.data[i]
        test_X = test_dataset.data[i]
        train_size = len(train_X)
        test_size = len(test_X)
        train_X = train_X.view(train_size, -1)
        test_X = test_X.view(test_size, -1)
        if train_X.size(1) < dim:
            dim = train_X.size(1)
            
        pca = PCA(n_components=dim, svd_solver='full', random_state=config.seed)
        train_low_X = pca.fit_transform(train_X)
        test_low_X = pca.fit_transform(test_X)
        train_conatenatings.append(train_low_X)
        test_conatenatings.append(test_low_X)
        print(f"Evaluate view {i+1}")
        
        # print('Clustering:')
        
        # start_time = time()
        # acc, nmi, ari = single_view_clustering(np.r_[train_low_X, test_low_X], np.r_[train_Y, test_Y])
        # end_time = time()
        # print(f"clustering acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, times: {end_time-start_time:.2f}s")
        
        # print('Classification:')
        # start_time = time()
        # acc, p, fscore = single_view_classification(train_low_X, train_Y, test_low_X, test_Y)
        # end_time = time()
        # print(f"classification acc: {acc:.4f}, p: {p:.4f}, fscore: {fscore:.4f}, times: {end_time-start_time:.2f}s")
        
        # print(f'{"-"*60}')
    
    
    train_conatenatings = np.concatenate(train_conatenatings, axis=1)
    test_conatenatings = np.concatenate(test_conatenatings, axis=1)
    print(f"Evaluate concatenating")
    
    print('Clustering:')
    start_time = time()
    acc, nmi, ari = single_view_clustering(np.r_[train_conatenatings, test_conatenatings], 
                                           np.r_[train_Y, test_Y])
    end_time = time()
    print(f"clustering acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, times: {end_time-start_time:.2f}s")
    
    print('Classification:')
    start_time = time()
    acc, p, fscore = single_view_classification(train_conatenatings, train_Y, 
                                                test_conatenatings, test_Y)
    end_time = time()
    print(f"classification acc: {acc:.4f}, p: {p:.4f}, fscore: {fscore:.4f}, times: {end_time-start_time:.2f}s")
    


if __name__ == '__main__':
    main()