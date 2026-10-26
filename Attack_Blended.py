'''
This is the test code of poisoned training under Blended.
'''

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip
from PIL import Image

import core


# ========== ResNet-18_CIFAR-10_Blended ==========
dataset = torchvision.datasets.CIFAR10

global_seed = 666
deterministic = False
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

CUDA_SELECTED_DEVICES = '0'
datasets_root_dir = './data'

image_path = './resource/blended/hello_kitty.jpeg'
trigger_img = Image.open(image_path).convert('RGB').resize((32, 32))
trigger_array = np.asarray(trigger_img)
pattern = torch.from_numpy(trigger_array.copy()).permute(2, 0, 1).to(torch.uint8)
weight = torch.full((3, 32, 32), fill_value=0.2, dtype=torch.float32)

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

blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=0,
    poisoned_rate=0.1,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
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
    'experiment_name': 'CIFAR-10_Blended'
}
blended.train(schedule)