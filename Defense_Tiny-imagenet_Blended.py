import os
import time
import random
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Subset
import numpy as np
import core
from core.defenses.epar import EPAR
from PIL import Image


global_seed = 666
torch.manual_seed(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

CUDA_VISIBLE_DEVICES = '0'
datasets_root_dir = './data/tiny-imagenet-200'

TARGET_CLASS = 0
NUM_CLASSES = 200
TRUSTED_PER_CLASS = 10
CLASS_SIZE_FACTOR = 0.6

dataset = torchvision.datasets.ImageFolder
train_root = os.path.join(datasets_root_dir, 'train')
test_root = os.path.join(datasets_root_dir, 'val_classified')

transform_analysis = Compose([
    ToTensor()
])

transform_retrain = Compose([
    ToTensor()
])

transform_test = Compose([
    ToTensor()
])

analysis_trainset = dataset(train_root, transform=transform_analysis)
retrain_trainset = dataset(train_root, transform=transform_retrain)
testset = dataset(test_root, transform=transform_test)

assert len(analysis_trainset) == 100000
assert len(retrain_trainset) == 100000
assert len(testset) == 10000
assert len(analysis_trainset.classes) == NUM_CLASSES
assert analysis_trainset.class_to_idx == retrain_trainset.class_to_idx
assert analysis_trainset.class_to_idx == testset.class_to_idx
assert analysis_trainset.samples == retrain_trainset.samples

POISONED_CKPT_PATH = './experiments/Tiny-ImageNet_Blended/ckpt_epoch_100.pth'
POISONED_RATE = 0.1

image_path = './resource/blended/hello_kitty.jpeg'

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Blended trigger image not found: {image_path}")

trigger_img = Image.open(image_path).convert('RGB').resize((64, 64))
trigger_array = np.asarray(trigger_img)
pattern = (
    torch.from_numpy(trigger_array.copy())
    .permute(2, 0, 1)
    .to(torch.uint8)
)

weight = torch.full(
    (3, 64, 64),
    fill_value=0.2,
    dtype=torch.float32
)

attack_analysis = core.Blended(
    train_dataset=analysis_trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=NUM_CLASSES),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=TARGET_CLASS,
    poisoned_rate=POISONED_RATE,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    seed=global_seed,
    deterministic=False
)

p_train, p_test_full = attack_analysis.get_poisoned_dataset()

attack_retrain = core.Blended(
    train_dataset=retrain_trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=NUM_CLASSES),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=TARGET_CLASS,
    poisoned_rate=POISONED_RATE,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    seed=global_seed,
    deterministic=False
)

p_retrain, _ = attack_retrain.get_poisoned_dataset()

assert p_train.poisoned_set == p_retrain.poisoned_set
print(f"Poisoned training samples: {len(p_train.poisoned_set)}")

trusted_indices = []
rng = np.random.default_rng(global_seed)

train_targets = np.array(analysis_trainset.targets)
poisoned_indices = set(p_train.poisoned_set)

for c in range(NUM_CLASSES):
    class_indices = np.where(train_targets == c)[0].tolist()
    class_indices = [i for i in class_indices if i not in poisoned_indices]
    rng.shuffle(class_indices)
    trusted_indices.extend(class_indices[:TRUSTED_PER_CLASS])

trusted_indices = sorted(trusted_indices)
trusted_clean_set = Subset(analysis_trainset, trusted_indices)

clean_testset = testset

test_targets = np.array(testset.targets)
asr_indices = [
    i for i in range(len(testset))
    if test_targets[i] != TARGET_CLASS
]
p_test = Subset(p_test_full, asr_indices)

model_poisoned = core.models.ResNet(18, num_classes=NUM_CLASSES)

if os.path.exists(POISONED_CKPT_PATH):
    print(f"Loading poisoned model from {POISONED_CKPT_PATH}...")
    state_dict = torch.load(POISONED_CKPT_PATH, map_location='cpu')

    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    new_state_dict = {
        k[7:] if k.startswith('module.') else k: v
        for k, v in state_dict.items()
    }
    model_poisoned.load_state_dict(new_state_dict)


else:
    raise FileNotFoundError(f"Checkpoint not found: {POISONED_CKPT_PATH}")

cul_schedule = {
    'batch_size': 128,
    'num_workers': 8,
    'epochs': 20,
    'lr': 0.01,
    'schedule': [15],
    'shots': TRUSTED_PER_CLASS,
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
    'test_epoch_interval': 2
}

timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
exp_name = f'ResNet18_TinyImageNet_Blended_EPAR_{timestamp}'

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'save_dir': 'experiments/EPAR-defense',
    'experiment_name': exp_name,
    'dataset_name': 'TinyImageNet',
    'attack_name': 'Blended',
    'cache_tag': f'TinyImageNet_Blended_seed{global_seed}',
    'class_size_factor': CLASS_SIZE_FACTOR,
    'cul_schedule': cul_schedule,
    'retrain_schedule': retrain_schedule,
}

print(f"\n========== Running EPAR Defense: {exp_name} ==========")

defense = EPAR(
    model=model_poisoned,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=p_train,
    poisoned_testset=p_test,
    clean_testset=clean_testset,
    poisoned_retrainset=p_retrain,
    trusted_clean_set=trusted_clean_set,
    seed=global_seed,
    num_classes=NUM_CLASSES,
    target_class=TARGET_CLASS
)

model_factory = lambda: core.models.ResNet(18, num_classes=NUM_CLASSES)

defense.train(schedule, model_factory)
