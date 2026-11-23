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
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor
from PIL import Image

import core
import core.defenses.analysis_utils as utils
from core.defenses.epar import EPAR
from core.utils.accuracy import filter_poisoned_testset


def _robust_find_classes(directory):
    directory = osp.expanduser(directory)
    subdirs = [d.name for d in os.scandir(directory) if d.is_dir()]
    wnid_dirs = sorted([d for d in subdirs if len(d) == 9 and d.startswith('n') and d[1:].isdigit()])
    if wnid_dirs:
        return wnid_dirs, {cls_name: i for i, cls_name in enumerate(wnid_dirs)}
    annot_path = osp.join(directory, 'val_annotations.txt')
    if osp.exists(annot_path):
        sibling_train = osp.join(osp.dirname(directory), 'train')
        if osp.isdir(sibling_train):
            train_wnids = sorted([
                d.name for d in os.scandir(sibling_train)
                if d.is_dir() and len(d.name) == 9 and d.name.startswith('n')
            ])
            if train_wnids:
                return train_wnids, {cls_name: i for i, cls_name in enumerate(train_wnids)}
        with open(annot_path, 'r') as f:
            classes = sorted({line.strip().split('\t')[1] for line in f if len(line.strip().split('\t')) >= 2})
        return classes, {cls_name: i for i, cls_name in enumerate(classes)}
    classes = sorted(subdirs)
    return classes, {cls_name: i for i, cls_name in enumerate(classes)}


def _robust_make_dataset(directory, class_to_idx=None, extensions=None, is_valid_file=None, allow_empty=True, **kwargs):
    directory = osp.expanduser(directory)
    if class_to_idx is None:
        _, class_to_idx = _robust_find_classes(directory)
    valid_exts = ('.jpeg', '.jpg', '.png', '.bmp', '.ppm', '.pgm', '.tif', '.tiff', '.webp')
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = osp.join(directory, target_class)
        if not osp.isdir(target_dir):
            continue
        img_dir = osp.join(target_dir, 'images') if osp.isdir(osp.join(target_dir, 'images')) else target_dir
        for root, _, fnames in sorted(os.walk(img_dir, followlinks=True)):
            for fname in sorted(fnames):
                if fname.lower().endswith(valid_exts):
                    instances.append((osp.join(root, fname), class_index))
    if not instances:
        annot_path = osp.join(directory, 'val_annotations.txt')
        raw_img_dir = osp.join(directory, 'images')
        if osp.exists(annot_path) and osp.isdir(raw_img_dir):
            with open(annot_path, 'r') as f:
                annot_map = {
                    line.strip().split('\t')[0]: line.strip().split('\t')[1]
                    for line in f if len(line.strip().split('\t')) >= 2
                }
            for fname in sorted(os.listdir(raw_img_dir)):
                if fname.lower().endswith(valid_exts) and fname in annot_map:
                    wnid = annot_map[fname]
                    if wnid in class_to_idx:
                        instances.append((osp.join(raw_img_dir, fname), class_to_idx[wnid]))
    return instances


DatasetFolder.find_classes = lambda self, directory: _robust_find_classes(directory)
DatasetFolder.make_dataset = staticmethod(_robust_make_dataset)


class RobustTinyImageNetFolder(torchvision.datasets.ImageFolder):
    def find_classes(self, directory):
        return _robust_find_classes(directory)

    @staticmethod
    def make_dataset(directory, class_to_idx=None, extensions=None, is_valid_file=None, allow_empty=True, **kwargs):
        return _robust_make_dataset(
            directory, class_to_idx=class_to_idx, extensions=extensions,
            is_valid_file=is_valid_file, allow_empty=allow_empty, **kwargs
        )


RobustTinyImageNetFolder.__name__ = 'ImageFolder'


def gen_wanet_grids(img_size=64, k=4, s=0.5, seed=666):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = nn.functional.interpolate(ins, size=img_size, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=img_size)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    torch.set_rng_state(rng_state)
    return identity_grid, noise_grid * s


parser = argparse.ArgumentParser(description="Tiny-ImageNet EPAR Defense")
parser.add_argument('--attack', type=str, default='WaNet',
                    choices=['BadNets', 'Blended', 'Refool', 'WaNet'])
parser.add_argument('--ckpt_path', type=str, default='./experiments/Tiny-ImageNet_WaNet_2026-11-19_17:44:35/ckpt_epoch_100.pth')
parser.add_argument('--seed', type=int, default=777)
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

val_dir = osp.join(datasets_root_dir, 'val_classified')
if not osp.isdir(val_dir):
    val_dir = osp.join(datasets_root_dir, 'val')

transform = Compose([ToTensor()])
trainset = RobustTinyImageNetFolder(osp.join(datasets_root_dir, 'train'), transform=transform)
testset = RobustTinyImageNetFolder(val_dir, transform=transform)

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
    reflection_dir = './resource/refool/Refool-SelectedReflectionImages/selected_out-images'
    attack = core.Refool(
        train_dataset=trainset, test_dataset=testset, model=base_model, loss=loss_fn,
        y_target=TARGET_CLASS, poisoned_rate=POISONED_RATE,
        reflection_dir=reflection_dir, save_dir='experiments',
        seed=global_seed, deterministic=False
    )

elif attack_name == 'WaNet':
    ig_p = osp.join(ckpt_dir, 'identity_grid.pth')
    ng_p = osp.join(ckpt_dir, 'noise_grid.pth')
    if not osp.exists(ig_p):
        ig_p = osp.join(ckpt_dir, 'identity_grid.pt')
        ng_p = osp.join(ckpt_dir, 'noise_grid.pt')
    if osp.exists(ig_p) and osp.exists(ng_p):
        identity_grid = torch.load(ig_p, map_location='cpu')
        noise_grid = torch.load(ng_p, map_location='cpu')
    else:
        identity_grid, noise_grid = gen_wanet_grids(img_size=IMG_SIZE, k=4, s=0.5, seed=global_seed)
    attack = core.WaNet(
        train_dataset=trainset, test_dataset=testset, model=base_model, loss=loss_fn,
        y_target=TARGET_CLASS, poisoned_rate=POISONED_RATE,
        identity_grid=identity_grid, noise_grid=noise_grid, noise=True,
        poisoned_transform_train_index=0, poisoned_transform_test_index=0,
        poisoned_target_transform_index=0, seed=global_seed, deterministic=False
    )

p_train, p_test = attack.get_poisoned_dataset()
p_test = filter_poisoned_testset(p_test, testset, target_class=TARGET_CLASS)

trusted_clean_set = utils.get_few_shot_clean_subset(
    p_train, num_classes=NUM_CLASSES, shots=SHOTS_PER_CLASS, seed=global_seed
)

model_poisoned = core.models.ResNet(18, num_classes=NUM_CLASSES)
state_dict = torch.load(ckpt_path, map_location='cpu')
state_dict = state_dict.get('state_dict', state_dict.get('model_state_dict', state_dict))
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