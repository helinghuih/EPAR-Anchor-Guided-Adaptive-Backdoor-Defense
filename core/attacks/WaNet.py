
import copy
from copy import deepcopy
import random
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn as nn
from torchvision.transforms import Compose

from torchvision.datasets import CIFAR10, MNIST, DatasetFolder, GTSRB
from .base import *


class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img, noise=False):
        if noise:
            ins = torch.rand(1, self.h, self.h, 2) * self.noise_rescale - 1
            grid = self.grid + ins / self.h
            grid = torch.clamp(grid, -1, 1) # Note: grid already has correct shape
        else:
            grid = self.grid
        poison_img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze(0)
        return poison_img


class AddDatasetFolderTrigger(AddTrigger):
    def __init__(self, identity_grid, noise_grid, noise, noise_rescale=2.0):
        super(AddDatasetFolderTrigger, self).__init__()
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid
        self.noise = noise
        self.noise_rescale = noise_rescale

        self.h = identity_grid.shape[1]
        self.grid = self.identity_grid + self.noise_grid / self.h
        self.grid = torch.clamp(self.grid, -1, 1)


    def __call__(self, img):
        is_pil = False
        if isinstance(img, PIL.Image.Image):
            is_pil = True
            img = F.to_tensor(img)

        if isinstance(img, torch.Tensor):
            img = F.convert_image_dtype(img, torch.float)
            img = self.add_trigger(img, noise=self.noise)

        if is_pil:
            img = F.to_pil_image(img)
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target
    def __call__(self, y_target):
        return self.y_target


class PoisonedDatasetFolder(DatasetFolder):
    def __init__(
        self,
        benign_dataset,
        y_target,
        poisoned_rate,
        identity_grid,
        noise_grid,
        noise,
        poisoned_transform_index,
        poisoned_target_transform_index,
        poisoned_set=None,
    ):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None,
        )

        total_num = len(benign_dataset)

        if poisoned_set is None:
            poisoned_num = int(total_num * poisoned_rate)
            if not 0 <= poisoned_num <= total_num:
                raise ValueError(
                    f"Invalid poisoned sample count: {poisoned_num}"
                )
            candidate_indices = list(range(total_num))
            random.shuffle(candidate_indices)
            selected_indices = candidate_indices[:poisoned_num]
        else:
            selected_indices = [int(index) for index in poisoned_set]
            invalid_indices = [
                index
                for index in selected_indices
                if index < 0 or index >= total_num
            ]
            if invalid_indices:
                raise ValueError(
                    "poisoned_set contains invalid indices: "
                    f"{invalid_indices[:10]}"
                )

        self.poisoned_set = frozenset(selected_indices)
        self.y_target = int(y_target)

        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        self.poisoned_transform.transforms.insert(
            poisoned_transform_index,
            AddDatasetFolderTrigger(
                identity_grid,
                noise_grid,
                noise,
            ),
        )

        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(
                self.target_transform
            )

        self.poisoned_target_transform.transforms.insert(
            poisoned_target_transform_index,
            ModifyTarget(y_target),
        )

        self.poisoned_targets = [
            self.y_target if index in self.poisoned_set else int(target)
            for index, (_, target) in enumerate(self.samples)
        ]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target

class PoisonedCIFAR10(CIFAR10):
    def __init__(self, benign_dataset, y_target, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index):
        super(PoisonedCIFAR10, self).__init__(benign_dataset.root, benign_dataset.train, benign_dataset.transform, benign_dataset.target_transform, download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        if self.transform is None: self.poisoned_transform = Compose([])
        else: self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(identity_grid, noise_grid, noise))

        if self.target_transform is None: self.poisoned_target_transform = Compose([])
        else: self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None: img = self.transform(img)
            if self.target_transform is not None: target = self.target_transform(target)
        return img, target


class PoisonedMNIST(MNIST):
    def __init__(self, benign_dataset, y_target, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index):
        super(PoisonedMNIST, self).__init__(benign_dataset.root, benign_dataset.train, benign_dataset.transform, benign_dataset.target_transform, download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        if self.transform is None: self.poisoned_transform = Compose([])
        else: self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(identity_grid, noise_grid, noise))

        if self.target_transform is None: self.poisoned_target_transform = Compose([])
        else: self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None: img = self.transform(img)
            if self.target_transform is not None: target = self.target_transform(target)
        return img, target

class PoisonedGTSRB(GTSRB):
    def __init__(self, benign_dataset, y_target, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index):
        super(PoisonedGTSRB, self).__init__(
            benign_dataset.root,
            split=benign_dataset._split,
            transform=benign_dataset.transform,
            target_transform=benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        if self.transform is None: self.poisoned_transform = Compose([])
        else: self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(identity_grid, noise_grid, noise))

        if self.target_transform is None: self.poisoned_target_transform = Compose([])
        else: self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        path, target = self._samples[index]
        sample = Image.open(path).convert("RGB")
        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None: sample = self.transform(sample)
            if self.target_transform is not None: target = self.target_transform(target)
        return sample, target


def CreatePoisonedDataset(
    benign_dataset,
    y_target,
    poisoned_rate,
    identity_grid,
    noise_grid,
    noise,
    poisoned_transform_index,
    poisoned_target_transform_index,
    poisoned_set=None,
):

    if isinstance(benign_dataset, GTSRB):
        return PoisonedGTSRB(
            benign_dataset,
            y_target,
            poisoned_rate,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_index,
            poisoned_target_transform_index,
        )

    if isinstance(benign_dataset, CIFAR10):
        return PoisonedCIFAR10(
            benign_dataset,
            y_target,
            poisoned_rate,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_index,
            poisoned_target_transform_index,
        )

    if isinstance(benign_dataset, MNIST):
        return PoisonedMNIST(
            benign_dataset,
            y_target,
            poisoned_rate,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_index,
            poisoned_target_transform_index,
        )

    if isinstance(benign_dataset, DatasetFolder):
        return PoisonedDatasetFolder(
            benign_dataset,
            y_target,
            poisoned_rate,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_index,
            poisoned_target_transform_index,
            poisoned_set=poisoned_set,
        )

    raise NotImplementedError(
        f"Unsupported dataset type: {type(benign_dataset)}"
    )

class WaNet(Base):
    """Construct poisoned datasets with WaNet method."""
    def __init__(self, train_dataset, test_dataset, model, loss, y_target, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_train_index=0, poisoned_transform_test_index=0, poisoned_target_transform_index=0, schedule=None, seed=0, deterministic=False):
        super(WaNet, self).__init__(train_dataset, test_dataset, model, loss, schedule, seed, deterministic)
        self.poisoned_train_dataset = CreatePoisonedDataset(train_dataset, y_target, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_train_index, poisoned_target_transform_index)
        self.poisoned_test_dataset = CreatePoisonedDataset(test_dataset, y_target, 1.0, identity_grid, noise_grid, noise, poisoned_transform_test_index, poisoned_target_transform_index)