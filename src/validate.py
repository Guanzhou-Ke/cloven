"""
Validation.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/11/18

"""
import argparse
from time import time

import torch
import numpy as np
from torch.utils.data import DataLoader

from experiments import models_cls
from config import get_cfg_defaults
from datatool import load_dataset
from utils import (clustering_by_representation, 
                   clustering_metric,
                   print_network, 
                   classify_via_svm,
                   classify_via_vote,
                   get_masked)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, help='path to config file')
    parser.add_argument('--model', '-m', type=str, help='path to model file')
    args = parser.parse_args()
    return args


def get_predicts(model, dataloader, device, views, masks=None):
    ground_truth = []
    predicts = []
    for data in dataloader:
        Xs, y = data
        if masks is not None:
            # masked data
            for idx in range(len(masks)):
                Xs[idx] = Xs[idx].masked_fill(masks[idx], 0)
        Xs = [Xs[idx].to(device) for idx in range(views)]
        ground_truth.append(y)
        preds = model.predict(Xs)
        predicts.append(preds)
    ground_truth = torch.concat(ground_truth, dim=-1).numpy()
    predicts = torch.concat(predicts, dim=-1).squeeze().detach().cpu().numpy()
    return predicts, ground_truth


def get_representation(model, dataloader, device, views, masks=None):
    ground_truth = []
    representations = []
    for data in dataloader:
        Xs, y = data
        if masks is not None:
            # masked data
            for idx in range(len(masks)):
                Xs[idx] = Xs[idx].masked_fill(masks[idx], 0)
        Xs = [Xs[idx].to(device) for idx in range(views)]
        ground_truth.append(y)
        repre = model.commonZ(Xs)
        representations.append(repre)
    ground_truth = torch.concat(ground_truth, dim=-1).numpy()
    representations = torch.vstack(representations).squeeze().detach().cpu().numpy()
    return representations, ground_truth


def validate_clustering(model, train_loader, test_loader, device, views, masks=None):
    """For clustering, we use all data.

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        test_loader (_type_): _description_
        device (_type_): _description_

    Returns:
        metrics tuple
    """
    if model.can_predict:
        train_predicts, train_ground_truth = get_predicts(model, train_loader, device, views, masks=masks)
        test_predicts, test_ground_truth = get_predicts(model, test_loader, device, views, masks=masks)
        predicts = np.r_[train_predicts, test_predicts]
        ground_truth = np.r_[train_ground_truth, test_ground_truth]
        start_time = time()
        acc, nmi, ari, class_acc, p, fscore = clustering_metric(ground_truth, predicts)
        end_time = time()
        print(f"Consuming {end_time-start_time:.2f}s")
    else:
        train_representations, train_ground_truth = get_representation(model, train_loader, device, views, masks=masks)
        test_representations, test_ground_truth = get_representation(model, test_loader, device, views, masks=masks)
        representations = np.r_[train_representations, test_representations]
        ground_truth = np.r_[train_ground_truth, test_ground_truth]
        start_time = time()
        acc, nmi, ari, class_acc, p, fscore = clustering_by_representation(representations, ground_truth)
        end_time = time()
        print(f"Consuming {end_time-start_time:.2f}s")
    return acc, nmi, ari


def validate_classification(model, train_dataloader, test_dataloader, device, views, masks=None):
    train_X, train_Y = get_representation(model, train_dataloader, device, views, masks=masks)
    test_X, test_Y = get_representation(model, test_dataloader, device, views, masks=masks)
    start_time = time()
    acc, p, fscore = classify_via_svm(train_X, train_Y, test_X, test_Y)
    end_time = time()
    print(f"Consuming {end_time-start_time:.2f}s")
    return acc, p, fscore


def validate_har(model, train_dataloader, test_dataloader, device, views, masks=None):
    train_X, train_Y = get_representation(model, train_dataloader, device, views, masks=masks)
    test_X, test_Y = get_representation(model, test_dataloader, device, views, masks=masks)
    acc, p, fscore = classify_via_vote(train_X, train_Y, test_X, test_Y)
    return acc, p, fscore


def validate(model, args, train_dataset, test_dataset, device='cpu', masks=None):
    model.eval()
    train_loader = DataLoader(train_dataset, 
                             args.train.batch_size, 
                             args.train.num_workers,
                             drop_last=True if masks is not None else False)
    test_loader = DataLoader(test_dataset, 
                             args.train.batch_size, 
                             args.train.num_workers,
                             drop_last=True if masks is not None else False)
    print(f"{'-'*30} Validation {'-'*30}")
    if 'clustering' in args.task:
        print(f'Validation clustering ....')
        clustering_acc, nmi, ari = validate_clustering(model, train_loader, test_loader, device, args.views, masks=masks)
        print(f'[Kmeans] Clustering ACC: {clustering_acc}, NMI: {nmi}, ARI: {ari}')
    if 'classification' in args.task:
        print(f'Validation classification ....')
        classification_acc, p, fscore = validate_classification(model, train_loader, test_loader, device, args.views, masks=masks)
        print(f'[SVM] Classification ACC: {classification_acc}, P: {p}, fscore: {fscore}\n')
    if args.dataset.name in ['DHA23', 'UWA30']:
        print(f'Validation Human Action Recognition ....')
        classification_acc, p, fscore = validate_har(model, train_loader, test_loader, device, args.views, masks=masks)
        print(f'[VOTE] Classification ACC: {classification_acc}, P: {p}, fscore: {fscore}\n')
    

def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg_defaults()
    config.merge_from_file(args.file)
    # set all split as true for validation
    config.dataset.split = True
    config.freeze()
    device = torch.device(f"cuda:{config.device}") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Use: {device}')

    # Select model class.
    model_cls = models_cls.get(config.model_name, None)
    if model_cls is None:
        raise ValueError(f'Cannot identify model name`{config.model_name}`. Please sure it has added in model dict.')
    model = model_cls(config, device=device)
    model.load_state_dict(torch.load(args.model), strict=False)
    print_network(model)
    model = model.to(device, non_blocking=True)
    # Load dataset
    train_dataset, test_dataset = load_dataset(config)
    # get mask.
    masked = False if config.train.missing_ratio == 0 else True
    if masked:
        print(f"Mask rate: {config.train.missing_ratio}")
        masks = get_masked(config.train.batch_size, 
                           config.backbone.input_shapes,
                           config.train.missing_ratio)
    else:
        masks = None
    print(f"Training dataset: {len(train_dataset)}, test dataset: {len(test_dataset)}")
    # Validate Model.
    validate(model, config, train_dataset, test_dataset, device=device, masks=masks)


if __name__ == '__main__':
    main()