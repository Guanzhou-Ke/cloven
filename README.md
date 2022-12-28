# A Clustering-guided Contrastive Fusion for Multi-view Representation Learning
The official repos. for "A Clustering-guided Contrastive Fusion for Multi-view Representation Learning".

- Submitted at: IEEE Transaction on Image Process
- Status: Submitted.


## Dataset

The Scene-15 datasets are placed in `src/data/processed` folder.

The EdgeMNIST and FashionMNIST datasets could automatically download and generate by initialization.

The COIL-20 and COIL-100 datasets could be downloaded by [this url](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php). After that, please put them into `src/data/raw` directory.

Note: the `src/datatool.py` can automatically process the raw data like MNIST, FashionMNIST, COIL-20, and COIL-100.


## Validation

### For evaluate the multi-view model.

Run:
```
cd src
python validate.py -f <path to config file> -m <path to model file>
```

Note: The checkpoints of CLOVEN could be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1QQbb_uW9E0mYu-NxMCWZ7w) **password: pema**

### For evaluate the single-view methods.

Please run `src/singleview.py`


## Training

Our original model placed at `src/experiments/algorithms/cloven.py`.

Train the model by running following command:

```
python main.py -f <path to your config>
```

The `main.py` could automatically find the model class and dataset that defined in config file.

## Advance experiment

You can modify the configuration files at `./src/experiments/configs`. We employ the [YACS](https://github.com/rbgirshick/yacs) style to write the experimental configs.

Example:

```yaml
# For YourAlg
# the number of views
views: 3
# the dimensionality of the view-specific encoder's outputs.
hidden_dim: 512
# how many times for training your alg.
test_time: 10
# train on which gpu device if gpu=True.
device: 1
# random seed for reproducibility
seed: 0
# set your algorithm here.
experiment: 'youralg'
# experiment id
experiment_id: 0
# print training logs.
verbose: true
note: 'experiment note'
# view-specific encoders' backbone, support mlp-like and cnn-like networks. see the networks module.
backbone:
  type: 'cnn'
  encoders: [
      'resnet18', 'resnet18', 'resnet18'
  ]
  kernals: [
      [7, 5, 3, 3, 3],
      [7, 5, 3, 3, 3]
  ]
  # for cnn networks, define the input's channel.
  channels: 1
  # share view-specific encoder if true.
  shared: true
  # [channel, H, W]
  # input shapes for recontruction.
  input_shapes: [
    [1, 224, 224],
    [1, 224, 224],
    [1, 224, 224]
  ]
# dataset setting.
dataset:
  name: 'coil-20'
  # Resize the image's size.
  imsize: 224
# Training setting. Please see the configs.py
train:
  epochs: 100
  batch_size: 64
  optim: "adam"
  fp16: true
  lr: 0.001
  evaluate: 1
  save_embeddings: -1
  missing_ratio: 0.
  enable_auto_weight: false
```

## Common issues

- (Apex) The issue of "tuple index out of range" from cached_cast.

Modify the `apex/amp/utils.py#cached_cast` as following:

```
# change this line (line 113)
- if cached_x.grad_fn.next_functions[1][0].variable is not x:
# into this
+ if cached_x.grad_fn.next_functions[0][0].variable is not x:
```

[Issue Link](https://github.com/NVIDIA/apex/issues/694)