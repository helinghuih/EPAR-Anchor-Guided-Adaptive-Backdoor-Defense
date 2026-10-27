'''
This is the test code of EPAR defense on Tiny-ImageNet.
'''

import argparse
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
import core.defenses.analysis_utils as utils
from core.defenses.epar import EPAR
from core.utils.accuracy import filter_poisoned_testset


parser = argparse.ArgumentParser(description="Tiny-ImageNet EPAR Defense")
parser.add_argument('--attack', type=str, default='BadNets',
                    choices=['BadNets', 'Blended', 'Refool'])
parser.add_argument('--ckpt_path', type=str, default='./experiments/Tiny-ImageNet_BadNets/ckpt_epoch_100.pth')
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

global_seed = args.seed
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

CUDA_SELECTED_DEVICES = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_SELECTED_DEVICES
datasets_root_dir = './data/tiny-imagenet-200'

TARGET_CLASS = 0
NUM_CLASSES = 200
SHOTS_PER_CLASS = 10
IMG_SIZE = 64
POISONED_RATE = 0.1

transform = Compose([ToTensor()])
trainset = torchvision.datasets.ImageFolder(osp.join(datasets_root_dir, 'train'), transform=transform)
testset = torchvision.datasets.ImageFolder(osp.join(datasets_root_dir, 'val'), transform=transform)

base_model = core.models.ResNet(18, num_classes=NUM_CLASSES)
loss_fn = nn.CrossEntropyLoss()
attack_name = args.attack
ckpt_path = args.ckpt_path
ckpt_dir = osp.dirname(ckpt_path)

if attack_name == 'BadNets':
    pattern = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    pattern[:, -6:, -6:] = 255
    weight = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    weight[:, -6:, -6:] = 1.0
    attack = core.BadNets(
        train_dataset=trainset, test_dataset=testset, model=base_model, loss=loss_fn,
        pattern=pattern, weight=weight, y_target=TARGET_CLASS, poisoned_rate=POISONED_RATE,
        poisoned_transform_train_index=0, poisoned_transform_test_index=0,
        poisoned_target_transform_index=0, seed=global_seed, deterministic=False
    )

elif attack_name == 'Blended':
    trigger_img = Image.open('./resource/blended/hello_kitty.jpeg').convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    trigger_array = np.asarray(trigger_img)
    pattern = torch.from_numpy(trigger_array.copy()).permute(2, 0, 1).to(torch.uint8)
    weight = torch.full((3, IMG_SIZE, IMG_SIZE), fill_value=0.2, dtype=torch.float32)
    attack = core.Blended(
        train_dataset=trainset, test_dataset=testset, model=base_model, loss=loss_fn,
        pattern=pattern, weight=weight, y_target=TARGET_CLASS, poisoned_rate=POISONED_RATE,
        poisoned_transform_train_index=0, poisoned_transform_test_index=0,
        poisoned_target_transform_index=0, seed=global_seed, deterministic=False
    )

elif attack_name == 'Refool':
    reflection_dir = './resource/refool/Refool-SelectedReflectionImages/selected_out-of-focus'
    attack = core.Refool(
        train_dataset=trainset, test_dataset=testset, model=base_model, loss=loss_fn,
        y_target=TARGET_CLASS, poisoned_rate=POISONED_RATE,
        reflection_dir=reflection_dir, save_dir='experiments',
        seed=global_seed, deterministic=False
    )

p_train, p_test = attack.get_poisoned_dataset()
p_test = filter_poisoned_testset(p_test, testset, target_class=TARGET_CLASS)

trusted_clean_set = utils.get_few_shot_clean_subset(
    p_train, num_classes=NUM_CLASSES, shots=SHOTS_PER_CLASS, seed=global_seed
)

model_poisoned = core.models.ResNet(18, num_classes=NUM_CLASSES)
state_dict = torch.load(ckpt_path, map_location='cpu')
new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
model_poisoned.load_state_dict(new_state_dict)

cul_schedule = {
    'batch_size': 128,
    'num_workers': 4,
    'epochs': 20,
    'lr': 1e-05,
    'weight_decay': 5e-4,
    'schedule': [20, 20],
    'max_unlearn_loss': 7.0,
    'shots': SHOTS_PER_CLASS,
}

retrain_schedule = {
    'batch_size': 128,
    'num_workers': 8,
    'epochs': 100,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [75, 90],
    'test_epoch_interval': 5,
    'use_retrain_aug': False,
    'transform': False
}

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_SELECTED_DEVICES,
    'save_dir': 'experiments/EPAR-defense',
    'experiment_name': f'ResNet18_TinyImageNet_{attack_name}',
    'dataset_name': 'TinyImageNet',
    'attack_name': attack_name,
    'cache_tag': f'TinyImageNet_{attack_name}_seed{global_seed}',
    'drop_ratio': 0.4,
    'cul_schedule': cul_schedule,
    'retrain_schedule': retrain_schedule,
}

defense = EPAR(
    model=model_poisoned,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=p_train,
    poisoned_testset=p_test,
    clean_testset=testset,
    poisoned_retrainset=p_train,
    trusted_clean_set=trusted_clean_set,
    seed=global_seed,
    num_classes=NUM_CLASSES,
    target_class=TARGET_CLASS
)

model_factory = lambda: core.models.ResNet(18, num_classes=NUM_CLASSES)
defense.train(schedule, model_factory)