'''
This is the test code of poisoned training under WaNet.
'''

import os.path as osp
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop

import core


# ========== ResNet-18_CIFAR-10_WaNet ==========
dataset = torchvision.datasets.CIFAR10

global_seed = 666
deterministic = True
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

CUDA_SELECTED_DEVICES = '0'
datasets_root_dir = './data'


def gen_grid(height, k):
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return identity_grid, noise_grid


transform_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

identity_grid, noise_grid = gen_grid(32, 4)

wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [75, 90],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'CIFAR-10_WaNet'
}
wanet.train(schedule)

ckpt_dirs = [d for d in glob.glob(osp.join('experiments', 'CIFAR-10_WaNet_*')) if osp.isdir(d)]
if ckpt_dirs:
    ckpt_dir = max(ckpt_dirs, key=osp.getmtime)
    torch.save(identity_grid, osp.join(ckpt_dir, 'identity_grid.pth'))
    torch.save(noise_grid, osp.join(ckpt_dir, 'noise_grid.pth'))