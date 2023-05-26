"""
Run observations.
---
Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
"""
DEFAULT_DATA_ROOT = '/home/hades/notebooks/Experiment/RIM-CAC/src/data/raw/'

import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from torchvision.models.resnet import resnet18

from utils import seed_everything, classify_via_svm, clustering_by_representation, get_masked


def edge_transformation(img):
    """
    edge preprocess functuin.
    """
    trans = transforms.Compose([image_edge,transforms.ToPILImage()])
    return trans(img)


def image_edge(img):
    """
    :param img:
    :return:
    """
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return edge


def get_transforms():
    return transforms.Compose([transforms.ToTensor(), 
                               transforms.Normalize(mean=[[0.1307]], std=[[0.3081]])])
    


class EdgeMNISTDataset(torchvision.datasets.MNIST):
    """
    """
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 download=False,
                 views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        
        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return [view0, view1], self.targets[idx]
    
    
class DCLLoss(torch.nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    SMALL_NUM = np.log(1e-45)
    
    def forward(self, z1, z2):
        return self.get_loss(z1, z2)
    
    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCLLoss, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
    

    def get_loss(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * self.SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()
    
    
class ResNet18(nn.Module):
    
    def __init__(self, channels=1):
        super(ResNet18, self).__init__()
        model = resnet18(pretrained=False)
        self.f = []
        for name, module in model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return feature, F.normalize(out, p=2)
    
    
class ToyNet(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet18()
        self.contrast_criterion = DCLLoss(temperature=0.1)
        
    def get_loss(self, x1, x2):
        _, out1 = self(x1)
        _, out2 = self(x2)

        loss = (self.contrast_criterion(out1, out2) + self.contrast_criterion(out2, out1)) / 2
        return loss
    
    def forward(self, x):
        return self.backbone(x)
    
    
def distance_metric(x1, x2):
    """Pair-wise distance."""
    return torch.cdist(x1, x2, p=2)


def train():
    seed = 42
    target_label = 7
    sample_nums = 100
    seed_everything(seed)
    trans = get_transforms()
    train_dataset = EdgeMNISTDataset(DEFAULT_DATA_ROOT, train=True, transform=trans)
    test_dataset = EdgeMNISTDataset(DEFAULT_DATA_ROOT, train=False, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=512, drop_last=True, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)
    
    index = test_dataset.targets == target_label
    target_index = np.argwhere(index == True).tolist()[0]
    targets_set_v0 = []
    targets_set_v1 = []
    for _ in range(sample_nums):
        (v0, v1), l = test_dataset[target_index[_]]
        targets_set_v0.append(v0)
        targets_set_v1.append(v1)
    targets_set_v0 = torch.stack(targets_set_v0).cuda()
    targets_set_v1 = torch.stack(targets_set_v1).cuda()
    
    epochs = 500
    # corresponding to distance 12, 7, 6
    save_epoch = [1, 50]
    distances = []
    accs = []
    net = ToyNet().cuda()
    optim = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(1, epochs+1):
        net.train()
        lossess = []
        for data, _ in train_loader:
            data = [x.cuda(non_blocking=True) for x in data]
            x1, x2 = data
            loss = net.get_loss(x1, x2)
        
            optim.zero_grad()
            loss.backward()
            lossess.append(loss.item())
            optim.step()
        print(f"Epoch: [{epoch}/{epochs}]  -   Training loss: {np.mean(lossess): .6f}")    
        net.eval()
        gts = []
        reprs = []
        with torch.no_grad():
            # for data, labels in test_loader:
            #     data = [x.cuda(non_blocking=True) for x in data]
            #     x1, x2 = data
            #     gts.append(labels)
            #     reprs.append(torch.cat([x1, x2], dim=1))
            # gts = torch.concat(gts, dim=-1).numpy()
            # reprs = torch.vstack(reprs).squeeze().detach().cpu().numpy()
            
            feas1, _ = net(targets_set_v0)
            feas2, _ = net(targets_set_v1)
            dist = distance_metric(feas1, feas2)
            print(f"Epoch: {epoch}, target label: {target_label}, dist: {dist.mean(): .6f}")
            distances.append(dist)
        if epoch in save_epoch:
            torch.save(net.state_dict(), f'./observation_model_ep_{epoch}.pth')
    torch.save(distances, './distances_t0.1.pth')
    torch.save(net.state_dict(), f'./observation_model_ep_{500}.pth')


def observation1():
    # 12, 7, 6
    # model_path = './observation_model_ep_1.pth'
    # model_path = './observation_model_ep_50.pth'
    model_path = './observation_model_ep_500.pth'
    seed = 42
    target_label = 7
    sample_nums = 100
    seed_everything(seed)
    trans = get_transforms()
    train_dataset = EdgeMNISTDataset(DEFAULT_DATA_ROOT, train=True, transform=trans)
    test_dataset = EdgeMNISTDataset(DEFAULT_DATA_ROOT, train=False, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=512, drop_last=False, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, pin_memory=True)
    
    index = test_dataset.targets == target_label
    target_index = np.argwhere(index == True).tolist()[0]
    targets_set_v0 = []
    targets_set_v1 = []
    for _ in range(sample_nums):
        (v0, v1), l = test_dataset[target_index[_]]
        targets_set_v0.append(v0)
        targets_set_v1.append(v1)
    targets_set_v0 = torch.stack(targets_set_v0).cuda()
    targets_set_v1 = torch.stack(targets_set_v1).cuda()
    accs = []
    net = ToyNet()
    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
    net.eval()
    
    print(f"run t-sne")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    
    targets_set_x = torch.cat([targets_set_v0, targets_set_v1])
    with torch.no_grad():
        targets_set_x, _ = net(targets_set_x)
    targets_set_x = tsne.fit_transform(targets_set_x.detach().cpu().numpy())
    torch.save(targets_set_x, './dis_6_emb.pth')
    for miss_rate in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        seed_everything(seed)
        masks = get_masked(10000, [(1, 28, 28), (1, 28, 28)], miss_rate)
        print(masks[0].shape)
        with torch.no_grad():
            # train_feats = []
            # train_tagets = []
            # for data, targets in train_loader:
            #     data = [x.cuda(non_blocking=True) for x in data]
            #     x1, x2 = data
            #     x1, _ = net(x1)
            #     x2, _ = net(x2)
            #     train_feats.append(torch.cat([x1, x2], dim=1))
            #     train_tagets.append(targets)
            # train_feats = torch.cat(train_feats).detach().cpu().numpy()
            # train_tagets = torch.cat(train_tagets).detach().cpu().numpy()
            
            for data, targets in test_loader:
                data = [x.masked_fill(m, 1) for m, x in zip(masks, data)]
                data = [x.cuda(non_blocking=True) for x in data]
                x1, x2 = data
                x1, _ = net(x1)
                x2, _ = net(x2)
            # test_feats = torch.cat([x1, x2], dim=1).detach().cpu().numpy()
            test_feats = (x1+x2).detach().cpu().numpy()
            test_tagets = targets.numpy()
            
            
        
        print(test_feats.shape, test_tagets.shape)
        
        acc_clu, nmi, ari, _, _, _ = clustering_by_representation(test_feats, test_tagets)
        # acc_cls, p, fscore = classify_via_svm(train_feats, train_tagets, test_feats, test_tagets)
        
        accs.append(acc_clu)
        print(accs)
    torch.save(accs, f'./dis_6_acc.pth')
    
    

def visualize():
    from matplotlib import pyplot as plt
    dis_12_acc = torch.load('./dis_12_acc.pth')
    dis_7_acc = torch.load('./dis_7_acc.pth')
    dis_6_acc = torch.load('./dis_6_acc.pth')
    dis_12_emb = torch.load('./dis_12_emb.pth')
    dis_7_emb = torch.load('./dis_7_emb.pth')
    dis_6_emb = torch.load('./dis_6_emb.pth')
    
    dis_emb = dis_12_emb
    
    plt.scatter(dis_emb[:50, 0], dis_emb[:50, 1], label='view 1', alpha=0.9, marker='1')
    plt.scatter(dis_emb[50:, 0], dis_emb[50:, 1], label='view 2', alpha=0.9, marker='1')
    
    # plt.plot(dis_12_acc)
    # dis_6_acc.sort()
    # print(dis_6_acc)
    plt.legend(fontsize=18, markerscale=2)
    plt.axis('off')
    plt.show()
                    

    
if __name__ == '__main__':
    # train()
    # observation1()
    visualize()
    
    # emnist = EdgeMNISTDataset(DEFAULT_DATA_ROOT, train=False)
    # from matplotlib import pyplot as plt
    # index = emnist.targets == 7
    # target_index = np.argwhere(index == True).tolist()[0]
    
    # (v0, v1), l = emnist[target_index[1]]
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(v0, cmap='gray')
    # ax2.imshow(v1, cmap='gray')
    # plt.show()
    
    # r18 = ResNet18()
    # x = torch.rand(10, 1, 28, 28)
    # feas, cont = r18(x)
    # print(feas.shape, cont.shape)
    
    # distances = torch.load('./distances_t0.1.pth')
    # distances = [dis.mean().cpu().numpy() for dis in distances]
    # plt.plot(distances)
    # plt.show()
    # print(distances[0], distances[100], distances[-1])