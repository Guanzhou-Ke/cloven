"""
Data preprocessing tools
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""

import os
import random

import cv2
import scipy.io as sio
from scipy import sparse
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import utils


DEFAULT_DATA_ROOT = './data'
PROCESSED_DATA_ROOT = os.path.join(DEFAULT_DATA_ROOT, 'processed')
RAW_DATA_ROOT = os.path.join(DEFAULT_DATA_ROOT, 'raw')


def export_dataset(name, views, labels):
    """
    Save dataset as .npz files
    :param name:
    :param views:
    :param labels:
    :return:
    """
    os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_ROOT, f"{name}.npz")
    npz_dict = {"labels": labels, "n_views": len(views)}
    for i, v in enumerate(views):
        npz_dict[f"view_{i}"] = v
    np.savez(file_path, **npz_dict)


def image_edge(img):
    """
    :param img:
    :return:
    """
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return np.stack((img, edge), axis=-1)


def _mnist(dataset_class):
    img_transforms = transforms.Compose([image_edge,
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
    train_dataset = dataset_class(root=RAW_DATA_ROOT, train=True,
                            download=True, transform=img_transforms)
    test_dataset = dataset_class(root=RAW_DATA_ROOT, train=False,
                            download=False, transform=img_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    train_data, train_labels = list(train_loader)[0]
    test_data, test_labels = list(test_loader)[0]
    data = torch.cat([train_data, test_data], dim=0)
    labels = torch.cat([train_labels, test_labels], dim=0)
    return data, labels


def emnist():
    data, labels = _mnist(torchvision.datasets.MNIST)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("emnist", views=views, labels=labels)


def fmnist():
    data, labels = _mnist(torchvision.datasets.FashionMNIST)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("fmnist", views=views, labels=labels)


def coil(n_objs=20):
    """
    Download: 
    https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
    
    1. coil-20: 
    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip
    
    
    2. coil-100:
    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip
    """
    from skimage.io import imread
    assert n_objs in [20, 100]
    data_dir = os.path.join(RAW_DATA_ROOT, f"coil-{n_objs}")
    img_size = (1, 128, 128) if n_objs == 20 else (3, 128, 128)
    n_imgs = 72
    n_views = 3

    n = (n_objs * n_imgs) // n_views

    views = []
    labels = []

    img_idx = np.arange(n_imgs)

    for obj in range(n_objs):
        obj_list = []
        obj_img_idx = np.random.permutation(img_idx).reshape(n_views, n_imgs // n_views)
        labels += (n_imgs // n_views) * [obj]

        for view, indices in enumerate(obj_img_idx):
            sub_view = []
            for i, idx in enumerate(indices):
                if n_objs == 20:
                    fname = os.path.join(data_dir, f"obj{obj + 1}__{idx}.png")
                    img = imread(fname)[None, ...]
                else:
                    fname = os.path.join(data_dir, f"obj{obj + 1}__{idx * 5}.png")
                    img = imread(fname)
                if n_objs == 100:
                    img = np.transpose(img, (2, 0, 1))
                sub_view.append(img)
            obj_list.append(np.array(sub_view))
        views.append(np.array(obj_list))
    views = np.array(views)
    views = np.transpose(views, (1, 0, 2, 3, 4, 5)).reshape(n_views, n, *img_size)
    labels = np.array(labels)
    export_dataset(f"coil-{n_objs}", views=views, labels=labels)


def _load_npz(name):
    return np.load(os.path.join(PROCESSED_DATA_ROOT, f"{name}.npz"))

def _load_mat(name):
    return sio.loadmat(os.path.join(PROCESSED_DATA_ROOT, f"{name}.mat"))


class MultiviewDataset(Dataset):

    def __init__(self, views, labels, transform=None):
        self.data = views
        self.targets = torch.LongTensor(labels)
        if self.targets.min() == 1:
            self.targets -= 1
        self.transform = transform
        self.num_view = len(self.data)

    def __getitem__(self, idx):
        views = [self.data[v][idx].float() for v in range(self.num_view)]
        
        if self.transform is not None:
            views = [self.transform(view) for view in views]
        return views, self.targets[idx]

    def __len__(self):
        return len(self.targets)
    

class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]


def load_dataset(args):
    # load 
    func_dict = {
        'Scene-15': __load_Scene15,
        'LandUse-21': __load_LandUse21,
        'NoisyMNIST': __load_NoisyMNIST,
        'Caltech101-20': __load_Caltech101_20,
        'EdgeMnist': __load_EdgeMnist,
        'FashionMnist': __load_FashionMnist,
        'coil-20': __load_coil,
        'coil-100': __load_coil,
        'DHA23': __load_dha23_or_uwa30,
        'UWA30': __load_dha23_or_uwa30,
    }
    func = func_dict.get(args.dataset.name, None)
    if func is None:
        raise ValueError("Dataset name error.")
    train_set, test_set = func(args)
    return train_set, test_set


def wrap_dataset(views, labels, transform=None, split=False, test_size=0.2):
    if split:
        # 8 : 2
        collects = train_test_split(*views, labels, test_size=test_size, random_state=42)
        train_labels, test_labels = collects[-2], collects[-1]
        # stride = 2, view1_train_set, view1_test_set, view2_train_set, view2_test_set .....
        train_set = collects[0:-2:2]
        test_set = collects[1:-2:2]
        return MultiviewDataset(train_set, train_labels, transform=transform), \
               MultiviewDataset(test_set, test_labels,transform=transform)
    else:
        dataset = MultiviewDataset(views, labels, transform=transform)
        return dataset, dataset


def get_transforms(args):
    name = args.dataset.name
     # Set transform for image dataset.
    if name in ['EdgeMnist', 'FashionMnist', 'coil-20', 'coil-100']:
        imsize = args.dataset.imsize
        if imsize:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor()
            ])
        else:
            # Don't transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
    else:
        transform = None
    return transform


def __load_Scene15(args):
    name = args.dataset.name
    mat = _load_mat(name)
    X = mat['X'][0]
    XV1 = X[0].astype('float32')
    XV2 = X[1].astype('float32')
    Y = np.squeeze(mat['Y'])
    views = [torch.from_numpy(XV1), torch.from_numpy(XV2)]
    transform = get_transforms(args)
    return wrap_dataset(views, Y, transform=transform, split=args.dataset.split, test_size=args.dataset.test_size)
    

def __load_LandUse21(args):
    name = args.dataset.name
    mat = _load_mat(name)
    train_x = []
    train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
    train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
    train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
    index = random.sample(range(train_x[0].shape[0]), 2100)
    XV1 = train_x[1][index]
    XV2 = train_x[2][index]
    Y = np.squeeze(mat['Y']).astype('int')[index]
    views = [torch.from_numpy(XV1), torch.from_numpy(XV2)]
    transform = get_transforms(args)
    return wrap_dataset(views, Y, transform=transform, split=args.dataset.split, test_size=args.dataset.test_size)


def __load_NoisyMNIST(args):
    name = args.dataset.name
    mat = _load_mat(name)
    tune = DataSet_NoisyMNIST(mat['XV1'], mat['XV2'], mat['tuneLabel'])
    test = DataSet_NoisyMNIST(mat['XTe1'], mat['XTe2'], mat['testLabel'])
    XV1 = np.concatenate([tune.images1, test.images1], axis=0)
    XV2 = np.concatenate([tune.images2, test.images2], axis=0)
    Y = np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])])
    views = [torch.from_numpy(XV1), torch.from_numpy(XV2)]
    transform = get_transforms(args)
    return wrap_dataset(views, Y, transform=transform, split=args.dataset.split, test_size=args.dataset.test_size)


def __load_Caltech101_20(args):
    name = args.dataset.name
    mat = _load_mat(name)
    X = mat['X'][0]
    x = X[3]
    XV1 = utils.normalize(x).astype('float32')
    x = X[4]
    XV2 = utils.normalize(x).astype('float32')
    Y = np.squeeze(mat['Y']).astype('int')
    views = [torch.from_numpy(XV1), torch.from_numpy(XV2)]
    transform = get_transforms(args)
    return wrap_dataset(views, Y, transform=transform, split=args.dataset.split, test_size=args.dataset.test_size)


def __load_EdgeMnist(args):
    file_path = os.path.join(PROCESSED_DATA_ROOT, "emnist.npz")
    if not os.path.exists(file_path):
        emnist()
    data = _load_npz('emnist')
    Y = data['labels']
    XV1 = data['view_0']
    XV2 = data['view_1']
    views = [torch.from_numpy(XV1), torch.from_numpy(XV2)]
    transform = get_transforms(args)
    return wrap_dataset(views, Y, transform=transform, split=args.dataset.split, test_size=args.dataset.test_size)
    

def __load_FashionMnist(args):
    file_path = os.path.join(PROCESSED_DATA_ROOT, "fmnist.npz")
    if not os.path.exists(file_path):
        fmnist()
    data = _load_npz('fmnist')
    Y = data['labels']
    XV1 = data['view_0']
    XV2 = data['view_1']
    views = [torch.from_numpy(XV1), torch.from_numpy(XV2)]
    transform = get_transforms(args)
    return wrap_dataset(views, Y, transform=transform, split=args.dataset.split, test_size=args.dataset.test_size)

def __load_coil(args):
    name = args.dataset.name
    n_objs = 20 if '20' in name else 100
    file_path = os.path.join(PROCESSED_DATA_ROOT, f"coil-{n_objs}.npz")
    if not os.path.exists(file_path):
        coil(n_objs)
    data = _load_npz(name)
    Y = data['labels']
    XV1 = data['view_0']
    XV2 = data['view_1']
    XV3 = data['view_2']
    views = [torch.from_numpy(XV1), torch.from_numpy(XV2), torch.from_numpy(XV3)]
    transform = get_transforms(args)
    return wrap_dataset(views, Y, transform=transform, split=args.dataset.split, test_size=args.dataset.test_size)
    

def __load_dha23_or_uwa30(args):
    """
    Human action recognition (HAR) dataset, 
    references to COMPLETER: https://github.com/XLearning-SCU/2022-TPAMI-DCP/blob/6c2e890407b86e752dcc1275bdd35e13d9e18cdb/utils/datasets.py#L221.
    raw dataset in: https://github.com/XLearning-SCU/2022-TPAMI-DCP/tree/main/data
    """
    name = args.dataset.name
    train_file_path = os.path.join(RAW_DATA_ROOT, 'HAR', f"{name}_total_train.csv")
    test_file_path = os.path.join(RAW_DATA_ROOT, 'HAR', f"{name}_total_test.csv")
    
    # Depth feature -> 110-dimension
    # RGB feature -> 3x2048 dimension
    feature_num1 = 110
    feature_num2 = 3 * 2048
    feature_num = feature_num1 + feature_num2
    num = 0
    train_XV1, train_XV2, test_XV1, test_XV2, train_label, test_label = [], [], [], [], [], []
    # load .csv file for training
    f = open(train_file_path, 'r')
    for i in f:
        row1 = i.rstrip().split(',')[:-1]
        row = [float(x) for x in row1]
        train_XV1.append(row[0: feature_num1])
        train_XV2.append(row[feature_num1: feature_num])
        train_label.append(row[feature_num: ])
    f.close()
    
    # load .csv file for test
    f = open(test_file_path, 'r')
    for i in f:
        row1 = i.rstrip().split(',')[:-1]
        row = [float(x) for x in row1]
        test_XV1.append(row[0: feature_num1])
        test_XV2.append(row[feature_num1: feature_num])
        test_label.append(row[feature_num: ])
    f.close()
    
    XV1 = np.r_[train_XV1, test_XV1]
    XV2 = np.r_[train_XV2, test_XV2]
    # Convert one-hot to label.
    Y = np.argmax(np.r_[train_label, test_label], axis=1)
    views = [torch.from_numpy(XV1), torch.from_numpy(XV2)]
    transform = get_transforms(args)
    return wrap_dataset(views, Y, transform=transform, split=args.dataset.split, test_size=args.dataset.test_size)
   
 
def generate_tiny_dataset(name, dataset, sample_num=100):
    """
    Tiny data set for T-SNE to visualize the representation's structure.
    Only support EdgeMNIST, FashionMNIST.
    """
    assert name in ['EdgeMnist', 'FashionMnist']
    y = dataset.targets.unique()
    x1s = []
    x2s = []
    ys = []

    for _ in y:
        idx = dataset.targets == _
        x1, x2, yy = dataset.data[0][idx, :], dataset.data[1][idx, :], dataset.targets[idx]
        x1, x2, yy = x1[:sample_num], x2[:sample_num], yy[:sample_num]
        x1s.append(x1)
        x2s.append(x2)
        ys.append(yy)
    
    x1s = torch.vstack(x1s)
    x2s = torch.vstack(x2s)
    ys = torch.concat(ys)
    
    tiny_dataset = {
        "x1": x1s,
        "x2": x2s,
        "y": ys
    }
    os.makedirs("./experiments/tiny-data/", exist_ok=True)
    torch.save(tiny_dataset, f'./experiments/tiny-data/{name}_tiny.plk')


if __name__ == '__main__':
    pass
    # generating emnist.
    # emnist()
    # fmnist()
    # from config import get_cfg_defaults
    # config = get_cfg_defaults()
    # config.dataset.name = 'UWA30'
    # # config.dataset.split = True
    # train, test = load_dataset(config)
    # print(len(train), len(test) if test else 0)
    # print(train.targets.unique())
    # v1, v2 = train[0][0]
    # y = train[0][1]
    # print(v1.shape, v2.shape, y)

    
            
        
        
