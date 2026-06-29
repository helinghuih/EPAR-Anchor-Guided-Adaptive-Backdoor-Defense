
import copy
import random
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
import cv2
from scipy import stats
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder, GTSRB
from .base import *

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target
    def __call__(self, y_target):
        return self.y_target

class AddTriggerMixin(object):
    def __init__(self, total_num, reflection_cadidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        super(AddTriggerMixin,self).__init__()
        self.reflection_candidates = reflection_cadidates
        self.max_image_size=max_image_size
        self.reflection_candidates_index = np.random.randint(0,len(self.reflection_candidates),total_num)
        self.alpha_bs = 1.-np.random.uniform(0.05,0.45,total_num) if alpha_b<0 else np.zeros(total_num)+alpha_b
        self.ghost_values = (np.random.uniform(0,1,total_num) < ghost_rate)
        if offset == (0,0):
            self.offset_xs = np.random.randint(3, 9, total_num)
            self.offset_ys = np.random.randint(3, 9, total_num)
        else:
            self.offset_xs = np.zeros((total_num,),np.int32) + offset[0]
            self.offset_ys = np.zeros((total_num,),np.int32) + offset[1]
        self.ghost_alpha = ghost_alpha
        self.ghost_alpha_switchs = np.random.uniform(0,1,total_num)
        self.ghost_alphas = np.random.uniform(0.15,0.5,total_num) if ghost_alpha < 0 else np.zeros(total_num)+ghost_alpha
        self.sigmas = np.random.uniform(1,5,total_num) if sigma<0 else np.zeros(total_num)+sigma
        self.atts = 1.08 + np.random.random(total_num)/10.0
        self.new_ws = np.random.uniform(0,1,total_num)
        self.new_hs = np.random.uniform(0,1,total_num)

    def _add_trigger(self, sample, index):
        # ... (与原文件一致) ...
        img_b = sample.permute(1,2,0).numpy()
        img_r = self.reflection_candidates[self.reflection_candidates_index[index]]
        h, w, channels = img_b.shape
        if channels == 1 and img_r.shape[-1]==3:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
        b = np.float32(img_b) / 255.
        r = np.float32(img_r) / 255.
        scale_ratio = float(max(h, w)) / float(self.max_image_size)
        w, h = (self.max_image_size, int(round(h / scale_ratio))) if w > h else (int(round(w / scale_ratio)), self.max_image_size)
        b = cv2.resize(b, (w, h), cv2.INTER_CUBIC)
        r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)
        if channels == 1:
            b = b[:,:,np.newaxis]
            r = r[:,:,np.newaxis]
        alpha_b = self.alpha_bs[index]
        if self.ghost_values[index]:
            b = np.power(b, 2.2)
            r = np.power(r, 2.2)
            offset = (self.offset_xs[index],self.offset_ys[index])
            r_1 = np.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)), 'constant', constant_values=0)
            r_2 = np.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)), 'constant', constant_values=(0, 0))
            ghost_alpha = self.ghost_alpha
            if ghost_alpha < 0:
                ghost_alpha_switch = 1 if self.ghost_alpha_switchs[index] > 0.5 else 0
                ghost_alpha = abs(ghost_alpha_switch - self.ghost_alphas[index])
            ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
            ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
            if channels==1:
                ghost_r = ghost_r[:,:,np.newaxis]
            reflection_mask = ghost_r * (1 - alpha_b)
            blended = reflection_mask + b * alpha_b
            transmission_layer = np.power(b * alpha_b, 1 / 2.2)
            ghost_r = np.power(reflection_mask, 1 / 2.2)
            ghost_r[ghost_r > 1.] = 1.
            ghost_r[ghost_r < 0.] = 0.
            blended = np.power(blended, 1 / 2.2)
            blended[blended > 1.] = 1.
            blended[blended < 0.] = 0.
            reflection_layer = np.uint8(ghost_r * 255)
            blended = np.uint8(blended * 255)
            transmission_layer = np.uint8(transmission_layer * 255)
        else:
            sigma = self.sigmas[index]
            b = np.power(b, 2.2)
            r = np.power(r, 2.2)
            sz = int(2 * np.ceil(2 * sigma) + 1)
            r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
            if channels==1:
                r_blur = r_blur[:,:,np.newaxis]
            blend = r_blur + b
            att = self.atts[index]
            for i in range(channels):
                maski = blend[:, :, i] > 1
                mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
                r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
            r_blur[r_blur >= 1] = 1
            r_blur[r_blur <= 0] = 0
            def gen_kernel(kern_len=100, nsig=1):
                interval = (2 * nsig + 1.) / kern_len
                x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
                kern1d = np.diff(stats.norm.cdf(x))
                kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
                kernel = kernel_raw / kernel_raw.sum()
                kernel = kernel / kernel.max()
                return kernel
            h, w = r_blur.shape[0: 2]
            new_w = int(self.new_ws[index]*(self.max_image_size - w - 10)) if w < self.max_image_size - 10 else 0
            new_h = int(self.new_hs[index]*(self.max_image_size - h - 10)) if h < self.max_image_size - 10 else 0
            g_mask = gen_kernel(self.max_image_size, 3)
            g_mask = np.dstack((g_mask, )*channels)
            alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_b / 2.)
            r_blur_mask = np.multiply(r_blur, alpha_r)
            blur_r = min(1., 4 * (1 - alpha_b)) * r_blur_mask
            blend = r_blur_mask + b * alpha_b
            transmission_layer = np.power(b * alpha_b, 1 / 2.2)
            r_blur_mask = np.power(blur_r, 1 / 2.2)
            blend = np.power(blend, 1 / 2.2)
            blend[blend >= 1] = 1
            blend[blend <= 0] = 0
            blended = np.uint8(blend * 255)
        return torch.from_numpy(blended).permute(2, 0, 1)

class AddDatasetFolderTriggerMixin(AddTriggerMixin):
    def add_trigger(self, img, index):
        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = self._add_trigger(img,index)
            if img.size(0) == 1: img = Image.fromarray(img.squeeze().numpy(), mode='L')
            elif img.size(0) == 3: img = Image.fromarray(img.permute(1, 2, 0).numpy())
            else: raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = self._add_trigger(img,index)
                img = img.numpy()
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = self._add_trigger(img,index)
                img = img.permute(1, 2, 0).numpy()
            return img
        elif type(img) == torch.Tensor:
            if img.dim() == 2: img = self._add_trigger(img,index)
            else:
                img = img.permute(2, 0, 1)
                img = self._add_trigger(img,index)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))

class AddMNISTTriggerMixin(AddTriggerMixin):
    def add_trigger(self, img, index):
        img = F.pil_to_tensor(img)
        img = self._add_trigger(img, index)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img

class AddCIFAR10TriggerMixin(AddTriggerMixin):
    def add_trigger(self, img, index):
        img = F.pil_to_tensor(img)
        img = self._add_trigger(img, index)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img

class PoisonedDatasetFolder(DatasetFolder, AddDatasetFolderTriggerMixin):
    """Refool-poisoned DatasetFolder/ImageFolder.

    ``poisoned_set`` is optional. When it is omitted, the original Refool
    behavior is retained: indices are selected by shuffling all dataset
    indices with Python's ``random`` module. Supplying it allows attack
    training and later defenses to reuse exactly the same poisoned indices.
    """

    def __init__(
        self,
        benign_dataset,
        y_target,
        poisoned_rate,
        poisoned_transform_index,
        poisoned_target_transform_index,
        reflection_cadidates,
        max_image_size=560,
        ghost_rate=0.49,
        alpha_b=-1.0,
        offset=(0, 0),
        sigma=-1,
        ghost_alpha=-1.0,
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
            tmp_list = list(range(total_num))
            random.shuffle(tmp_list)
            selected_indices = tmp_list[:poisoned_num]
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
            if len(set(selected_indices)) != len(selected_indices):
                raise ValueError("poisoned_set contains duplicate indices.")

        self.poisoned_set = frozenset(selected_indices)
        self.y_target = int(y_target)

        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        if poisoned_transform_index < 0:
            poisoned_transform_index = (
                len(self.poisoned_transform.transforms)
                + poisoned_transform_index
            )

        self.pre_poisoned_transform = Compose(
            self.poisoned_transform.transforms[:poisoned_transform_index]
        )
        self.post_poisoned_transform = Compose(
            self.poisoned_transform.transforms[poisoned_transform_index:]
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

        # Effective labels after poisoning. This is useful for ABL/NAD/REFINE
        # and does not change __getitem__ or the attack-training behavior.
        self.poisoned_targets = [
            self.y_target if index in self.poisoned_set else int(target)
            for index, (_, target) in enumerate(self.samples)
        ]

        AddDatasetFolderTriggerMixin.__init__(
            self,
            total_num,
            reflection_cadidates,
            max_image_size,
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha,
        )
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if index in self.poisoned_set:
            if len(self.pre_poisoned_transform.transforms): sample = self.pre_poisoned_transform(sample)
            sample = self.add_trigger(sample, index)
            sample = self.post_poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None: sample = self.transform(sample)
            if self.target_transform is not None: target = self.target_transform(target)
        return sample, target

class PoisonedCIFAR10(CIFAR10, AddCIFAR10TriggerMixin):
    def __init__(self, benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index, reflection_cadidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        super(PoisonedCIFAR10, self).__init__(benign_dataset.root, benign_dataset.train, benign_dataset.transform, benign_dataset.target_transform, download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        if self.transform is None: self.poisoned_transform = Compose([])
        else: self.poisoned_transform = copy.deepcopy(self.transform)
        if poisoned_transform_index < 0: poisoned_transform_index = len(self.poisoned_transform.transforms) + poisoned_transform_index
        self.pre_poisoned_transform = Compose(self.poisoned_transform.transforms[:poisoned_transform_index])
        self.post_poisoned_transform = Compose(self.poisoned_transform.transforms[poisoned_transform_index:])
        if self.target_transform is None: self.poisoned_target_transform = Compose([])
        else: self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        AddCIFAR10TriggerMixin.__init__(self, total_num, reflection_cadidates, max_image_size, ghost_rate, alpha_b, offset, sigma, ghost_alpha)
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if index in self.poisoned_set:
            if len(self.pre_poisoned_transform.transforms): img = self.pre_poisoned_transform(img)
            img = self.add_trigger(img, index)
            img = self.post_poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None: img = self.transform(img)
            if self.target_transform is not None: target = self.target_transform(target)
        return img, target

class PoisonedMNIST(MNIST, AddMNISTTriggerMixin):
    def __init__(self, benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index, reflection_cadidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        super(PoisonedMNIST, self).__init__(benign_dataset.root, benign_dataset.train, benign_dataset.transform, benign_dataset.target_transform, download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        if self.transform is None: self.poisoned_transform = Compose([])
        else: self.poisoned_transform = copy.deepcopy(self.transform)
        if poisoned_transform_index < 0: poisoned_transform_index = len(self.poisoned_transform.transforms) + poisoned_transform_index
        self.pre_poisoned_transform = Compose(self.poisoned_transform.transforms[:poisoned_transform_index])
        self.post_poisoned_transform = Compose(self.poisoned_transform.transforms[poisoned_transform_index:])
        if self.target_transform is None: self.poisoned_target_transform = Compose([])
        else: self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        AddMNISTTriggerMixin.__init__(self, total_num, reflection_cadidates, max_image_size, ghost_rate, alpha_b, offset, sigma, ghost_alpha)
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        if index in self.poisoned_set:
            if len(self.pre_poisoned_transform.transforms): img = self.pre_poisoned_transform(img)
            img = self.add_trigger(img, index)
            img = self.post_poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None: img = self.transform(img)
            if self.target_transform is not None: target = self.target_transform(target)
        return img, target

class PoisonedGTSRB(GTSRB, AddDatasetFolderTriggerMixin):
    def __init__(self, benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index, reflection_cadidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
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
        if poisoned_transform_index < 0: poisoned_transform_index = len(self.poisoned_transform.transforms) + poisoned_transform_index
        self.pre_poisoned_transform = Compose(self.poisoned_transform.transforms[:poisoned_transform_index])
        self.post_poisoned_transform = Compose(self.poisoned_transform.transforms[poisoned_transform_index:])

        if self.target_transform is None: self.poisoned_target_transform = Compose([])
        else: self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

        AddDatasetFolderTriggerMixin.__init__(self, total_num, reflection_cadidates, max_image_size, ghost_rate, alpha_b, offset, sigma, ghost_alpha)

    def __getitem__(self, index):
        path, target = self._samples[index]
        sample = Image.open(path).convert("RGB")
        if index in self.poisoned_set:
            if len(self.pre_poisoned_transform.transforms): sample = self.pre_poisoned_transform(sample)
            sample = self.add_trigger(sample, index)
            sample = self.post_poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None: sample = self.transform(sample)
            if self.target_transform is not None: target = self.target_transform(target)
        return sample, target

def CreatePoisonedDataset(
    benign_dataset,
    y_target,
    poisoned_rate,
    poisoned_transform_index,
    poisoned_target_transform_index,
    reflection_cadidates,
    max_image_size=560,
    ghost_rate=0.49,
    alpha_b=-1.0,
    offset=(0, 0),
    sigma=-1,
    ghost_alpha=-1.0,
    poisoned_set=None,
):


    if isinstance(benign_dataset, CIFAR10):
        if poisoned_set is not None:
            raise ValueError(
                "Shared poisoned_set is currently implemented for "
                "DatasetFolder/ImageFolder only."
            )
        return PoisonedCIFAR10(
            benign_dataset,
            y_target,
            poisoned_rate,
            poisoned_transform_index,
            poisoned_target_transform_index,
            reflection_cadidates,
            max_image_size,
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha,
        )

    if isinstance(benign_dataset, MNIST):
        if poisoned_set is not None:
            raise ValueError(
                "Shared poisoned_set is currently implemented for "
                "DatasetFolder/ImageFolder only."
            )
        return PoisonedMNIST(
            benign_dataset,
            y_target,
            poisoned_rate,
            poisoned_transform_index,
            poisoned_target_transform_index,
            reflection_cadidates,
            max_image_size,
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha,
        )

    if isinstance(benign_dataset, GTSRB):
        if poisoned_set is not None:
            raise ValueError(
                "Shared poisoned_set is currently implemented for "
                "DatasetFolder/ImageFolder only."
            )
        return PoisonedGTSRB(
            benign_dataset,
            y_target,
            poisoned_rate,
            poisoned_transform_index,
            poisoned_target_transform_index,
            reflection_cadidates,
            max_image_size,
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha,
        )

    if isinstance(benign_dataset, DatasetFolder):
        return PoisonedDatasetFolder(
            benign_dataset,
            y_target,
            poisoned_rate,
            poisoned_transform_index,
            poisoned_target_transform_index,
            reflection_cadidates,
            max_image_size,
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha,
            poisoned_set=poisoned_set,
        )

    raise NotImplementedError(
        f"Unsupported dataset type: {type(benign_dataset).__name__}"
    )

class Refool(Base):
    def __init__(
        self,
        train_dataset,
        test_dataset,
        model,
        loss,
        y_target,
        poisoned_rate,
        reflection_candidates,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        schedule=None,
        seed=0,
        deterministic=False,
        max_image_size=560,
        ghost_rate=0.49,
        alpha_b=-1.0,
        offset=(0, 0),
        sigma=-1,
        ghost_alpha=-1.0,
        poisoned_train_set=None,
        poisoned_test_set=None,
    ):


        super(Refool, self).__init__(
            train_dataset,
            test_dataset,
            model,
            loss,
            schedule,
            seed,
            deterministic,
        )

        self.poisoned_train_dataset = CreatePoisonedDataset(
            train_dataset,
            y_target,
            poisoned_rate,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
            reflection_candidates,
            max_image_size,
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha,
            poisoned_set=poisoned_train_set,
        )

        self.poisoned_test_dataset = CreatePoisonedDataset(
            test_dataset,
            y_target,
            1.0,
            poisoned_transform_test_index,
            poisoned_target_transform_index,
            reflection_candidates,
            max_image_size,
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha,
            poisoned_set=poisoned_test_set,
        )
