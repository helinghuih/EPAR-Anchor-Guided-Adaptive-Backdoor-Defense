import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

import core
import core.defenses.analysis_utils as utils
from core.defenses.epar import EPAR
from core.utils.accuracy import filter_poisoned_testset


global_seed = 666
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

datasets_root_dir = './data'
TARGET_CLASS = 0
POISONED_RATE = 0.1
SHOTS_PER_CLASS = 30
NUM_CLASSES = 10

ckpt_path = './experiments/CIFAR-10_Refool/ckpt_epoch_100.pth'
reflection_dir = './resource/refool/Refool-SelectedReflectionImages/selected_out-images'
EXP_DIR = './experiments/CIFAR-10_Refool_cache'
os.makedirs(EXP_DIR, exist_ok=True)

transform_train = Compose([
    Resize((32, 32)),
    ToTensor()
])
trainset = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    Resize((32, 32)),
    ToTensor()
])
testset = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=False, transform=transform_test, download=True)

retrain_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor()
])

attack = core.Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=TARGET_CLASS,
    poisoned_rate=POISONED_RATE,
    reflection_dir=reflection_dir,
    save_dir=EXP_DIR,
    seed=global_seed,
    deterministic=True
)

p_train, p_test = attack.get_poisoned_dataset()
p_test = filter_poisoned_testset(p_test, testset, target_class=TARGET_CLASS)

trusted_clean_set = utils.get_few_shot_clean_subset(
    p_train, num_classes=NUM_CLASSES, shots=SHOTS_PER_CLASS, seed=global_seed
)

model_poisoned = core.models.ResNet(18)
state_dict = torch.load(ckpt_path, map_location='cpu')
new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
model_poisoned.load_state_dict(new_state_dict)

cul_schedule = {
    'batch_size': 128,
    'num_workers': 4,
    'epochs': 20,
    'lr': 0.0001,
    'weight_decay': 5e-4,
    'schedule': [20, 20],
    'max_unlearn_loss': 7.0,
    'shots': SHOTS_PER_CLASS,
}

retrain_schedule = {
    'batch_size': 128,
    'num_workers': 4,
    'epochs': 100,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [75, 90],
    'transform': retrain_transform,
    'test_epoch_interval': 5
}

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'save_dir': 'experiments/EPAR-defense',
    'experiment_name': 'ResNet18_CIFAR10_Refool',
    'dataset_name': 'CIFAR10',
    'attack_name': 'Refool',
    'cache_tag': f'CIFAR10_Refool_seed{global_seed}',
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

model_factory = lambda: core.models.ResNet(18)
defense.train(schedule, model_factory)