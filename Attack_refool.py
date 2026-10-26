'''
This is the test code of poisoned training under Refool.
'''

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, Resize

import core


# ========== ResNet-18_CIFAR-10_Refool ==========
dataset = torchvision.datasets.CIFAR10

global_seed = 666
deterministic = True
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

CUDA_SELECTED_DEVICES = '0'
datasets_root_dir = './data'
REFLECTION_DATA_DIR = './resource/refool/Refool-SelectedReflectionImages/selected_out-images'
EXP_DIR = './experiments/CIFAR-10_Refool_cache'
os.makedirs(EXP_DIR, exist_ok=True)

transform_train = Compose([
    Resize((32, 32)),
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    Resize((32, 32)),
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

refool_attack = core.Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    reflection_dir=REFLECTION_DATA_DIR,
    save_dir=EXP_DIR,
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
    'experiment_name': 'CIFAR-10_Refool'
}
refool_attack.train(schedule)