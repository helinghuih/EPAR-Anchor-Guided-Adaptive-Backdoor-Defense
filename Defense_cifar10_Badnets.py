import os
import time
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip
import numpy as np
from PIL import Image
from core.utils.log import Log
import core
from core.defenses.epar import EPAR
from core.utils.accuracy import filter_poisoned_testset

global_seed = 666
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '0'
datasets_root_dir = './data'

POISONED_CKPT_PATH = './experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth'
TARGET_CLASS = 0

print("Initializing Data...")
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    RandomHorizontalFlip(),
    ToTensor()
])

trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([ToTensor()])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255

weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0

attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=TARGET_CLASS,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=True
)

p_train, p_test = attack.get_poisoned_dataset()
p_test = filter_poisoned_testset(p_test, testset, target_class=TARGET_CLASS)

model_poisoned = core.models.ResNet(18)
if os.path.exists(POISONED_CKPT_PATH):
    print(f"Loading poisoned model from {POISONED_CKPT_PATH}...")
    state_dict = torch.load(POISONED_CKPT_PATH, map_location='cpu')
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model_poisoned.load_state_dict(new_state_dict)
    print("Model loaded successfully.")
else:
    print(f"[Warning] Checkpoint not found at {POISONED_CKPT_PATH}. Using random init (Performance might be suboptimal).")

cul_schedule = {
    'batch_size': 128,
    'num_workers': 4,
    'epochs': 20,
    'lr': 0.01,
    'schedule': [15],
    'shots': 30,
}
retrain_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor()
])

retrain_schedule = {
    'batch_size': 128,
    'num_workers': 4,
    'epochs': 100,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [40, 70],
    'transform': retrain_transform,
    'test_epoch_interval': 2
}
timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
exp_name = f'ResNet18_CIFAR10_BadNets_EPAR_{timestamp}'

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'save_dir': 'experiments/EPAR-defense',
    'experiment_name': exp_name,
    'dataset_name': 'CIFAR10',
    'attack_name': 'Badnets',
    'cul_schedule': cul_schedule,
    'retrain_schedule': retrain_schedule,
}

print(f"\n========== Running EPAR Defense: {exp_name} ==========")
defense = EPAR(
    model=model_poisoned,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=p_train,
    poisoned_testset=p_test,
    clean_testset=testset,
    seed=global_seed,
    num_classes=10,
    target_class=0
)

model_factory = lambda: core.models.ResNet(18)

defense.train(schedule, model_factory)