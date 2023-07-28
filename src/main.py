import argparse

import torch

from experiments import models_cls
from config import get_cfg
from trainer import Trainer
from datatool import load_dataset, generate_tiny_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, help='Config File')
    parser.add_argument('--experiment', '-e', type=str, help='Experiment model name.')
    args = parser.parse_args()
    return args


def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args)
    device = torch.device(f"cuda:{config.device}") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Use: {device}')

    # Select model class.
    model_cls = models_cls.get(config.model_name, None)
    if model_cls is None:
        raise ValueError(f'Cannot identify model name`{config.model_name}`. Please sure it has added in model dict.')
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(config)
    if config.train.save_embeddings > 0:
        # generate tiny dataset for visualization latent space.
        generate_tiny_dataset(config.dataset.name, train_dataset)
    # Train Model.
    trainer = Trainer(model_cls, \
                      config, \
                      train_dataset, \
                      valid_dataset=test_dataset, \
                      device=device)
    trainer.train()


if __name__ == '__main__':
    main()
    