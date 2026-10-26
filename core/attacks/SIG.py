'''
This is the implement of SIG (Sinusoidal Signal Attack) [1].

Reference:
[1] A new backdoor attack in CNNs by training set corruption without label poisoning. ICIP 2019.
'''

import copy
import random

import numpy as np
import PIL
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder, GTSRB
from .base import *


class AddSIGTrigger:
    """Add sinusoidal signal trigger to image.

    Args:
        delta (float): Amplitude of the sinusoidal signal. Default: 20/255.0.
        f (int): Frequency of the sinusoidal signal. Default: 6.
    """

    def __init__(self, delta=20 / 255.0, f=6):
        self.delta = delta
        self.f = f

    def __call__(self, img):
        is_pil = False
        is_numpy = False

        if isinstance(img, PIL.Image.Image):
            is_pil = True
            tensor_img = F.to_tensor(img)  # [C, H, W] in [0.0, 1.0]
        elif isinstance(img, np.ndarray):
            is_numpy = True
            tensor_img = torch.from_numpy(img).float()
            if tensor_img.dim() == 3 and tensor_img.shape[-1] in (1, 3):  # HWC to CHW
                tensor_img = tensor_img.permute(2, 0, 1) / 255.0
            elif tensor_img.dtype == torch.uint8:
                tensor_img = tensor_img / 255.0
        elif isinstance(img, torch.Tensor):
            tensor_img = img.clone()
            if tensor_img.dtype == torch.uint8:
                tensor_img = tensor_img.float() / 255.0
        else:
            raise TypeError(
                "img should be PIL.Image.Image, numpy.ndarray, or torch.Tensor. "
                f"Got {type(img)}"
            )

        _, H, W = tensor_img.shape
        j = torch.arange(W, dtype=torch.float32).view(1, 1, W)
        pattern = self.delta * torch.sin(2 * np.pi * self.f * j / W)

        img_poisoned = torch.clamp(tensor_img + pattern, 0.0, 1.0)

        if is_pil:
            return F.to_pil_image(img_poisoned)
        elif is_numpy:
            output = (img_poisoned.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return output
        else:
            if getattr(img, 'dtype', None) == torch.uint8:
                return (img_poisoned * 255).to(torch.uint8)
            return img_poisoned


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, target):
        return self.y_target


class PoisonedDatasetFolder_SIG(DatasetFolder):
    def __init__(
            self,
            benign_dataset,
            y_target,
            poisoned_rate,
            delta,
            f,
            poisoned_transform_index,
            poisoned_target_transform_index,
            is_train=True,
    ):
        super(PoisonedDatasetFolder_SIG, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None,
        )

        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)

        if is_train:
            labels = np.array([s[1] for s in self.samples])
            target_indices = np.where(labels == y_target)[0]
            max_clean_label_poisons = int(len(target_indices) * 0.5)
            if poisoned_num > max_clean_label_poisons:
                print(
                    f"[SIG] Info: Capping clean-label poisons from {poisoned_num} to 50% of target class ({max_clean_label_poisons}/{len(target_indices)}).")
                poisoned_num = max_clean_label_poisons

            selected_indices = np.random.choice(target_indices, poisoned_num, replace=False).tolist()
        else:
            indices = list(range(total_num))
            random.shuffle(indices)
            selected_indices = indices[:poisoned_num]

        self.poisoned_set = frozenset(selected_indices)
        self.y_target = int(y_target)

        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        self.poisoned_transform.transforms.insert(
            poisoned_transform_index,
            AddSIGTrigger(delta=delta, f=f),
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


class PoisonedMNIST_SIG(MNIST):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 delta,
                 f,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 is_train=True):
        super(PoisonedMNIST_SIG, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)

        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)

        if is_train:
            labels = np.array(self.targets)
            target_indices = np.where(labels == y_target)[0]
            if poisoned_num > len(target_indices):
                poisoned_num = len(target_indices)
            selected_indices = np.random.choice(target_indices, poisoned_num, replace=False).tolist()
        else:
            tmp_list = list(range(total_num))
            random.shuffle(tmp_list)
            selected_indices = tmp_list[:poisoned_num]

        self.poisoned_set = frozenset(selected_indices)

        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddSIGTrigger(delta, f))

        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedCIFAR10_SIG(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 delta,
                 f,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 is_train=True):
        super(PoisonedCIFAR10_SIG, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)

        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)

        if is_train:
            labels = np.array(self.targets)
            target_indices = np.where(labels == y_target)[0]
            if poisoned_num > len(target_indices):
                poisoned_num = len(target_indices)
            selected_indices = np.random.choice(target_indices, poisoned_num, replace=False).tolist()
        else:
            tmp_list = list(range(total_num))
            random.shuffle(tmp_list)
            selected_indices = tmp_list[:poisoned_num]

        self.poisoned_set = frozenset(selected_indices)

        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddSIGTrigger(delta, f))

        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedGTSRB_SIG(GTSRB):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 delta,
                 f,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 is_train=True):
        super(PoisonedGTSRB_SIG, self).__init__(
            benign_dataset.root,
            split=benign_dataset._split,
            transform=benign_dataset.transform,
            target_transform=benign_dataset.target_transform,
            download=True)

        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)

        if is_train:
            labels = np.array([s[1] for s in self._samples])
            target_indices = np.where(labels == y_target)[0]
            max_clean_label_poisons = int(len(target_indices) * 0.5)
            if poisoned_num > max_clean_label_poisons:
                print(
                    f"[SIG] Info: Capping GTSRB clean-label poisons from {poisoned_num} to 50% of target class ({max_clean_label_poisons}/{len(target_indices)}).")
                poisoned_num = max_clean_label_poisons
            selected_indices = np.random.choice(target_indices, poisoned_num, replace=False).tolist()
        else:
            tmp_list = list(range(total_num))
            random.shuffle(tmp_list)
            selected_indices = tmp_list[:poisoned_num]

        self.poisoned_set = frozenset(selected_indices)

        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddSIGTrigger(delta, f))

        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        path, target = self._samples[index]
        sample = Image.open(path).convert("RGB")

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target


def CreatePoisonedDataset_SIG(
        benign_dataset,
        y_target,
        poisoned_rate,
        delta,
        f,
        poisoned_transform_index,
        poisoned_target_transform_index,
        is_train=True,
):
    if isinstance(benign_dataset, MNIST):
        return PoisonedMNIST_SIG(
            benign_dataset, y_target, poisoned_rate, delta, f,
            poisoned_transform_index, poisoned_target_transform_index, is_train
        )
    if isinstance(benign_dataset, CIFAR10):
        return PoisonedCIFAR10_SIG(
            benign_dataset, y_target, poisoned_rate, delta, f,
            poisoned_transform_index, poisoned_target_transform_index, is_train
        )
    if isinstance(benign_dataset, GTSRB):
        return PoisonedGTSRB_SIG(
            benign_dataset, y_target, poisoned_rate, delta, f,
            poisoned_transform_index, poisoned_target_transform_index, is_train
        )
    if isinstance(benign_dataset, DatasetFolder):
        return PoisonedDatasetFolder_SIG(
            benign_dataset, y_target, poisoned_rate, delta, f,
            poisoned_transform_index, poisoned_target_transform_index, is_train
        )
    raise NotImplementedError(f"Unsupported dataset type: {type(benign_dataset)}")


class SIG(Base):
    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 delta=20 / 255.0,
                 f=6,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False):
        super(SIG, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = CreatePoisonedDataset_SIG(
            train_dataset,
            y_target,
            poisoned_rate,
            delta,
            f,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
            is_train=True)

        self.poisoned_test_dataset = CreatePoisonedDataset_SIG(
            test_dataset,
            y_target,
            1.0,
            delta,
            f,
            poisoned_transform_test_index,
            poisoned_target_transform_index,
            is_train=False)