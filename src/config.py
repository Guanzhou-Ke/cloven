"""
Configuration loading tools.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import os

from yacs.config import CfgNode as CN


_C = CN()

# enable gpu
_C.gpu = True
# gpu device id.
_C.device = 0
# task type ['clustering', 'classification'], use `|` to add more task.
_C.task = None
# test time defaults to 5
_C.test_time = 5
# seed
_C.seed = 42
# print log
_C.verbose = True
# experiment name
_C.experiment = ''
# model name
_C.model_name = ''
_C.experiment_id = 0
# For multi-view setting
_C.views = None
# Network output dim.
_C.hidden_dim = 128
# Experiment Notes.
_C.note = ""


# Network setting.
_C.backbone = CN()
_C.backbone.type = None
# The setting of encoder layer, list.
_C.backbone.encoders = None
# Just for CNN len(encoders) - 2
_C.backbone.kernals = None
_C.backbone.activation = 'relu'
# put normalization layer before activation function.
_C.backbone.first_norm = False
_C.backbone.max_pooling = False
# Network initialize method, 'xavier' | 'gaussian' | 'kaiming' | 'orthogonal'
# default 'xavier'.
_C.backbone.init_method = 'xavier'
# enbale shared view-specific encoder
_C.backbone.shared = False
# set fc layer as identity, only enable if encoders were off-the-shelf network.
_C.backbone.fc_identity = True
# for cnn.
_C.backbone.channels = 1
# input shape  # List[ [channel, H, W] ]
_C.backbone.input_shapes = None
# decoder layers
_C.backbone.decoders = None
# for CNN decoder
_C.backbone.decoder_kernals = None
_C.backbone.decoder_strides = None
_C.backbone.decoder_output_paddings = None


# For dataset
_C.dataset = CN()
# ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST', 
# 'EdgeMnist', 'FashionMnist', 'coil-20', 'coil-100', 'DHA23', "UWA30"]
_C.dataset.name = 'Scene-15'
# for image.
_C.dataset.imsize = None
# split dataset
_C.dataset.split = False
# split ratio 8:2 (train:test)
_C.dataset.test_size = 0.2

# for training.
_C.train = CN()
_C.train.epochs = 100
_C.train.batch_size = None
_C.train.optim = None
_C.train.lr = 0.001
_C.train.num_workers = 4
_C.train.save_log = True
# if None, it will be set as './experiments/results/[model name]/[dataset name]'
_C.train.log_dir = None
# mix precision.
_C.train.fp16 = True
_C.train.opt_level = 'O1'
# Using ground truth
_C.train.with_gt = False
# For test embedding dataset
_C.train.save_embeddings = -1
# for incomplete setting, float. [0-1], 0 denotes complete.
_C.train.missing_ratio = 0.
# the interval of evaluate epoch, defaults to 5.
_C.train.evaluate = 5
# for reconstruction
_C.train.reconstruction = False
# Learning rate scheduler, [cosine, step]
_C.train.lr_scheduler = 'cosine'
# For training auto weight losses.
_C.train.enable_auto_weight = True
# full or decay', default='full' for adapted loss weight
_C.train.mean_sort = 'full'
# What decay to use with mean decay, default=1.0 for adapted loss weight
_C.train.mean_decay_param = 1.0


# For fusion
_C.fusion = CN()
# fusion type
_C.fusion.type = 'nn'
_C.fusion.activation = 'relu'
_C.fusion.use_bn = True
_C.fusion.num_layers = 2
# For self-attention
_C.fusion.nhead = 1
_C.fusion.batch_first = True
# enable position embedding.
_C.fusion.enable_pos = True
# self-attention feedforward layer dimension.
_C.fusion.attn_ffn_dim = 1024
# 
_C.fusion.norm_first = False
# aggregate method ['mean', 'sum', 'first', 'last']
_C.fusion.aggregate = 'mean'


# For clustering module
_C.cluster_module = CN()
# For ablation study.
_C.cluster_module.enable = True
_C.cluster_module.type = 'ddc'
_C.cluster_module.num_cluster = None
_C.cluster_module.cluster_hidden_dim = None # ddc hidden features

# For contrastive module
_C.contrastive = CN()
# For ablation study.
_C.contrastive.enable = True
_C.contrastive.ins_enable = True
_C.contrastive.cls_enable = True

_C.contrastive.type = 'simclr'
_C.contrastive.projection_dim = None
_C.contrastive.projection_hidden_dim = 512
# For simsiam
_C.contrastive.prediction_hidden_dim = 512
_C.contrastive.ins_lambda = 0.5
_C.contrastive.cls_lambda = 0.5
# For CONAN contrastive trade-off
_C.contrastive.con_lambda = 0.01
_C.contrastive.projection_layers = 2
_C.contrastive.temperature = 0.07
# For SwAV
_C.contrastive.nmb_protos = 256
_C.contrastive.eps = 0.05
_C.contrastive.ds_iters = 3
_C.contrastive.symmetry = False

# For inference
_C.inference = CN()
# for cca-based method's forward. if true, return h1, else return h2.
_C.inference.cca_return_h1 = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project.

    Returns:
        CfgNode: configuration.
    """
    return _C.clone()


def get_cfg(args):
    """Initialize configuration.
    """
    config = get_cfg_defaults()
    config.merge_from_file(args.file)
    if isinstance(config.task, str):
        config.task = config.task.split('|') if config.task else ['clustering', 'classification']
    config.model_name = config.experiment
    
    if not config.train.log_dir:
        path = f'./experiments/results/{config.experiment}/{config.dataset.name}/eid-{config.experiment_id}/'
        os.makedirs(path, exist_ok=True)
        config.train.log_dir = path
    else:
        os.makedirs(config.train.log_dir, exist_ok=True)
    config.freeze()
    return config



