'''
This is the test code of poisoned training on Tiny-ImageNet.
'''

import argparse
import glob
import os
import os.path as osp
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor
from PIL import Image

import core


def gen_wanet_grids(img_size=64, k=4, s=0.5, seed=666):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = nn.functional.upsample(ins, size=img_size, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=img_size)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    torch.set_rng_state(rng_state)
    return identity_grid, noise_grid * s


parser = argparse.ArgumentParser(description="Tiny-ImageNet Backdoor Attacks")
parser.add_argument('--attack', type=str, default='BadNets',
                    choices=['BadNets', 'Blended', 'Refool', 'WaNet'])
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

global_seed = args.seed
deterministic = False
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

CUDA_SELECTED_DEVICES = args.gpu
datasets_root_dir = './data/tiny-imagenet-200'
TARGET_CLASS = 0
NUM_CLASSES = 200
IMG_SIZE = 64
POISONED_RATE = 0.1

transform = Compose([ToTensor()])
trainset = torchvision.datasets.ImageFolder(osp.join(datasets_root_dir, 'train'), transform=transform)
testset = torchvision.datasets.ImageFolder(osp.join(datasets_root_dir, 'val'), transform=transform)

model = core.models.ResNet(18, num_classes=NUM_CLASSES)
loss_fn = nn.CrossEntropyLoss()
attack_name = args.attack
exp_name = f'Tiny-ImageNet_{attack_name}'

if attack_name == 'BadNets':
    pattern = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    pattern[:, -6:, -6:] = 255
    weight = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    weight[:, -6:, -6:] = 1.0

    attacker = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=loss_fn,
        y_target=TARGET_CLASS,
        poisoned_rate=POISONED_RATE,
        pattern=pattern,
        weight=weight,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        seed=global_seed,
        deterministic=deterministic,
    )

elif attack_name == 'Blended':
    trigger_img = Image.open('./resource/blended/hello_kitty.jpeg').convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    trigger_array = np.asarray(trigger_img)
    pattern = torch.from_numpy(trigger_array.copy()).permute(2, 0, 1).to(torch.uint8)
    weight = torch.full((3, IMG_SIZE, IMG_SIZE), fill_value=0.2, dtype=torch.float32)

    attacker = core.Blended(
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=loss_fn,
        pattern=pattern,
        weight=weight,
        y_target=TARGET_CLASS,
        poisoned_rate=POISONED_RATE,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        seed=global_seed,
        deterministic=deterministic,
    )

elif attack_name == 'Refool':
    reflection_dir = './resource/refool/Refool-SelectedReflectionImages/selected_out-of-focus'
    attacker = core.Refool(
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=loss_fn,
        y_target=TARGET_CLASS,
        poisoned_rate=POISONED_RATE,
        reflection_dir=reflection_dir,
        save_dir='experiments',
        seed=global_seed,
        deterministic=deterministic,
    )

elif attack_name == 'WaNet':
    identity_grid, noise_grid = gen_wanet_grids(img_size=IMG_SIZE, k=4, s=0.5, seed=global_seed)
    attacker = core.WaNet(
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=loss_fn,
        y_target=TARGET_CLASS,
        poisoned_rate=POISONED_RATE,
        identity_grid=identity_grid,
        noise_grid=noise_grid,
        noise=True,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        seed=global_seed,
        deterministic=deterministic,
    )

schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

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
    'experiment_name': exp_name,
}

attacker.train(schedule)

if attack_name == 'WaNet':
    exp_dirs = sorted(glob.glob(osp.join(schedule['save_dir'], f"{schedule['experiment_name']}_*")), key=osp.getmtime)
    if exp_dirs:
        torch.save(identity_grid, osp.join(exp_dirs[-1], 'identity_grid.pth'))
        torch.save(noise_grid, osp.join(exp_dirs[-1], 'noise_grid.pth'))