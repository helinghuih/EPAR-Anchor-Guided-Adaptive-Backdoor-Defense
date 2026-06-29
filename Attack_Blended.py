import os
import os.path as osp

import torch
import torch.nn as nn

import core
from core.utils.tiny_imagenet_utils import build_tiny_imagenet_views
from PIL import Image
import numpy as np


global_seed = 666
deterministic = False
torch.manual_seed(global_seed)

CUDA_SELECTED_DEVICES = "0"
datasets_root_dir = "./data/tiny-imagenet-200"

views = build_tiny_imagenet_views(datasets_root_dir)
trainset = views["train"]
testset = views["validation"]

assert len(trainset) == 100000
assert len(testset) == 10000
assert len(trainset.classes) == 200
assert trainset.class_to_idx == testset.class_to_idx

image_path = "./resource/blended/hello_kitty.jpeg"

if not osp.exists(image_path):
    raise FileNotFoundError(
        f"Blended trigger image not found: {image_path}"
    )

trigger_img = Image.open(image_path).convert("RGB").resize((64, 64))
trigger_array = np.asarray(trigger_img)
pattern = (
    torch.from_numpy(trigger_array.copy())
    .permute(2, 0, 1)
    .to(torch.uint8)
)

weight = torch.full(
    (3, 64, 64),
    fill_value=0.2,
    dtype=torch.float32,
)

blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=200),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=0,
    poisoned_rate=0.1,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    seed=global_seed,
    deterministic=deterministic,
)

poisoned_trainset, _ = blended.get_poisoned_dataset()
print(f"Blended poisoned training samples: {len(poisoned_trainset.poisoned_set)}")
schedule = {
    "device": "GPU",
    "CUDA_SELECTED_DEVICES": CUDA_SELECTED_DEVICES,

    "benign_training": False,
    "batch_size": 128,
    "num_workers": 8,

    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "gamma": 0.1,
    "schedule": [75, 90],
    "epochs": 100,

    "log_iteration_interval": 100,
    "test_epoch_interval": 10,
    "save_epoch_interval": 100,

    "save_dir": "experiments",
    "experiment_name": "Tiny-ImageNet_Blended",
}

blended.train(schedule)
blended.test(schedule)
