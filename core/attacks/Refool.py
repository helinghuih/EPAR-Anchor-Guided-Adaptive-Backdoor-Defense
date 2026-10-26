import os
import random
import torch
import numpy as np
import cv2
from PIL import Image
from scipy import stats
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from tqdm import tqdm
from .base import *


def read_image(img_path, type="RGB"):
    img = cv2.imread(img_path)
    if type is None:
        return img
    elif isinstance(type, str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type, str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError


class AddRefoolTrigger(object):
    """
    完全还原原版参数的 Refool 核心注入器
    """

    def __init__(self, total_num, reflection_dir, seed, max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0),
                 sigma=-1, ghost_alpha=-1.):
        self.max_image_size = max_image_size
        np.random.seed(seed)

        if not os.path.exists(reflection_dir):
            raise FileNotFoundError(f"Reflection images data {reflection_dir} not exist!")
        reflection_image_path = sorted(os.listdir(reflection_dir))
        self.reflection_candidates = [read_image(os.path.join(reflection_dir, img_path)) for img_path in
                                      reflection_image_path[:200]]

        self.reflection_candidates_index = np.random.randint(0, len(self.reflection_candidates), total_num)
        self.alpha_bs = 1. - np.random.uniform(0.05, 0.45, total_num) if alpha_b < 0 else np.zeros(total_num) + alpha_b
        self.ghost_values = (np.random.uniform(0, 1, total_num) < ghost_rate)

        if offset == (0, 0):
            self.offset_xs = np.random.randint(3, 9, total_num)
            self.offset_ys = np.random.randint(3, 9, total_num)
        else:
            self.offset_xs = np.zeros((total_num,), np.int32) + offset[0]
            self.offset_ys = np.zeros((total_num,), np.int32) + offset[1]

        self.ghost_alpha = ghost_alpha
        self.ghost_alpha_switchs = np.random.uniform(0, 1, total_num)
        self.ghost_alphas = np.random.uniform(0.15, 0.5, total_num) if ghost_alpha < 0 else np.zeros(
            total_num) + ghost_alpha
        self.sigmas = np.random.uniform(1, 5, total_num) if sigma < 0 else np.zeros(total_num) + sigma
        self.atts = 1.08 + np.random.random(total_num) / 10.0
        self.new_ws = np.random.uniform(0, 1, total_num)
        self.new_hs = np.random.uniform(0, 1, total_num)

    def __call__(self, img_b, index):
        img_b = img_b.astype(np.float32)
        img_r = self.reflection_candidates[self.reflection_candidates_index[index]]

        h, w, channels = img_b.shape
        if channels == 1 and img_r.shape[-1] == 3:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        b = img_b / 255.
        r = np.float32(img_r) / 255.

        scale_ratio = float(max(h, w)) / float(self.max_image_size)
        w_r, h_r = (self.max_image_size, int(round(h / scale_ratio))) if w > h else (int(round(w / scale_ratio)),
                                                                                     self.max_image_size)

        b = cv2.resize(b, (w_r, h_r), cv2.INTER_CUBIC)
        r = cv2.resize(r, (w_r, h_r), cv2.INTER_CUBIC)
        if channels == 1:
            b = b[:, :, np.newaxis]
            r = r[:, :, np.newaxis]

        alpha_b = self.alpha_bs[index]

        if self.ghost_values[index]:
            b = np.power(b, 2.2)
            r = np.power(r, 2.2)

            offset = (self.offset_xs[index], self.offset_ys[index])
            r_1 = np.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)), 'constant', constant_values=0)
            r_2 = np.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)), 'constant', constant_values=(0, 0))

            ghost_alpha = self.ghost_alpha
            if ghost_alpha < 0:
                ghost_alpha_switch = 1 if self.ghost_alpha_switchs[index] > 0.5 else 0
                ghost_alpha = abs(ghost_alpha_switch - self.ghost_alphas[index])

            ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
            ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w_r, h_r))
            if channels == 1:
                ghost_r = ghost_r[:, :, np.newaxis]

            reflection_mask = ghost_r * (1 - alpha_b)
            blended = reflection_mask + b * alpha_b

            blended = np.power(blended, 1 / 2.2)
            blended[blended > 1.] = 1.
            blended[blended < 0.] = 0.
            blended = np.uint8(blended * 255)
        else:
            sigma = self.sigmas[index]
            b = np.power(b, 2.2)
            r = np.power(r, 2.2)

            sz = int(2 * np.ceil(2 * sigma) + 1)
            r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
            if channels == 1:
                r_blur = r_blur[:, :, np.newaxis]

            att = self.atts[index]
            blend = r_blur + b
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

            new_w = int(self.new_ws[index] * (self.max_image_size - w_r - 10)) if w_r < self.max_image_size - 10 else 0
            new_h = int(self.new_hs[index] * (self.max_image_size - h_r - 10)) if h_r < self.max_image_size - 10 else 0

            g_mask = gen_kernel(self.max_image_size, 3)
            g_mask = np.dstack((g_mask,) * channels)
            alpha_r = g_mask[new_h: new_h + h_r, new_w: new_w + w_r, :] * (1. - alpha_b / 2.)

            r_blur_mask = np.multiply(r_blur, alpha_r)
            blend = r_blur_mask + b * alpha_b

            blend = np.power(blend, 1 / 2.2)
            blend[blend >= 1] = 1
            blend[blend <= 0] = 0
            blended = np.uint8(blend * 255)

        return Image.fromarray(blended)


class PrePoisonedDataset(Dataset):
    def __init__(self, data_path, prefix, transform=None, poisoned_rate=0.0, seed=0):
        img_path = os.path.join(data_path, f'{prefix}_imgs.pt')
        label_path = os.path.join(data_path, f'{prefix}_labels.pt')
        pset_path = os.path.join(data_path, f'{prefix}_poisoned_set.pt')

        self.imgs = torch.load(img_path, weights_only=False)
        self.labels = torch.load(label_path, weights_only=False)
        self.targets = [int(x) for x in self.labels]
        # Strip Resize from runtime transform since images in self.imgs are already resized
        if transform is not None and hasattr(transform, 'transforms'):
            filtered_t = [t for t in transform.transforms if not isinstance(t, Resize)]
            from torchvision.transforms import Compose
            self.transform = Compose(filtered_t)
        else:
            self.transform = transform

        if os.path.exists(pset_path):
            loaded_pset = torch.load(pset_path, weights_only=False)
            if isinstance(loaded_pset, torch.Tensor):
                self.poisoned_set = frozenset(loaded_pset.tolist())
            else:
                self.poisoned_set = frozenset(loaded_pset)
        else:
            total_num = len(self.labels)
            poisoned_num = int(total_num * poisoned_rate)
            rng = random.Random(seed)
            tmp_list = list(range(total_num))
            rng.shuffle(tmp_list)
            self.poisoned_set = frozenset(tmp_list[:poisoned_num])

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index].numpy())
        target = int(self.labels[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.labels)


def generate_offline_poisoned_data(dataset, y_target, poisoned_rate, reflection_dir, save_dir, seed, prefix="train"):
    os.makedirs(save_dir, exist_ok=True)
    img_save_path = os.path.join(save_dir, f'{prefix}_imgs.pt')
    label_save_path = os.path.join(save_dir, f'{prefix}_labels.pt')
    pset_save_path = os.path.join(save_dir, f'{prefix}_poisoned_set.pt')

    if os.path.exists(img_save_path) and os.path.exists(label_save_path):
        print(f"[*] {prefix} 离线投毒数据已存在于 {save_dir}，直接读取缓存。")
        return

    print(f"[*] 正在生成 {prefix} 离线投毒数据集 (兼容 CIFAR-10 / GTSRB / Tiny-ImageNet)...")

    is_imagefolder = not hasattr(dataset, 'data')
    samples_list = None

    if is_imagefolder:
        if hasattr(dataset, 'samples'):
            samples_list = dataset.samples
        elif hasattr(dataset, '_samples'):
            samples_list = dataset._samples
        else:
            raise AttributeError(f"Unsupported dataset without .data, .samples, or ._samples: {type(dataset)}")
        total_num = len(samples_list)
        targets = [int(s[1]) for s in samples_list]
    else:
        total_num = len(dataset.data)
        targets = [int(t) for t in dataset.targets]

    if hasattr(dataset, 'data'):
        target_size = (dataset.data.shape[1], dataset.data.shape[2])
    else:
        target_size = None
        if dataset.transform is not None:
            if isinstance(dataset.transform, Resize):
                ts = dataset.transform.size
                target_size = (ts, ts) if isinstance(ts, int) else tuple(ts)
            elif hasattr(dataset.transform, 'transforms'):
                for t in dataset.transform.transforms:
                    if isinstance(t, Resize):
                        ts = t.size
                        target_size = (ts, ts) if isinstance(ts, int) else tuple(ts)
                        break
        if target_size is None:
            with Image.open(samples_list[0][0]) as first_img:
                target_size = (first_img.height, first_img.width)

    poisoned_num = int(total_num * poisoned_rate)

    random.seed(seed)
    tmp_list = list(range(total_num))
    random.shuffle(tmp_list)
    poisoned_set = frozenset(tmp_list[:poisoned_num])

    injector = AddRefoolTrigger(total_num=total_num, reflection_dir=reflection_dir, seed=seed, max_image_size=560)
    downsample_transform = Resize(target_size)

    new_imgs = np.zeros((total_num, target_size[0], target_size[1], 3), dtype=np.uint8)
    new_labels = list(targets)

    for i in tqdm(range(total_num), desc=f"Processing {prefix}"):
        if is_imagefolder:
            img_path = samples_list[i][0]
            with Image.open(img_path) as raw_img:
                raw_img = raw_img.convert('RGB')
                raw_img = downsample_transform(raw_img)
                img = np.array(raw_img, dtype=np.uint8)
        else:
            img = dataset.data[i]

        if i in poisoned_set:
            img_pil_560 = injector(img, i)
            img_pil_target = downsample_transform(img_pil_560)
            new_imgs[i] = np.array(img_pil_target, dtype=np.uint8)
            new_labels[i] = int(y_target)
        else:
            new_imgs[i] = img

    torch.save(torch.from_numpy(new_imgs), img_save_path)
    torch.save(torch.LongTensor(new_labels), label_save_path)
    torch.save(torch.LongTensor(list(poisoned_set)), pset_save_path)
    print(f"[*] {prefix} 数据已保存至 {save_dir} (图像尺寸: {target_size})")


class Refool(Base):
    def __init__(self, train_dataset, test_dataset, model, loss, y_target, poisoned_rate,
                 reflection_dir, save_dir, schedule=None, seed=0, deterministic=False):
        super(Refool, self).__init__(
            train_dataset=train_dataset, test_dataset=test_dataset, model=model,
            loss=loss, schedule=schedule, seed=seed, deterministic=deterministic)

        dataset_tag = type(train_dataset).__name__
        poison_data_path = os.path.join(save_dir, f'poisoned_data_{dataset_tag}_target{y_target}_seed{seed}')

        generate_offline_poisoned_data(
            dataset=train_dataset, y_target=y_target, poisoned_rate=poisoned_rate,
            reflection_dir=reflection_dir, save_dir=poison_data_path, seed=seed, prefix="train"
        )
        generate_offline_poisoned_data(
            dataset=test_dataset, y_target=y_target, poisoned_rate=1.0,
            reflection_dir=reflection_dir, save_dir=poison_data_path, seed=seed, prefix="test"
        )

        self.poisoned_train_dataset = PrePoisonedDataset(
            data_path=poison_data_path, prefix="train", transform=train_dataset.transform,
            poisoned_rate=poisoned_rate, seed=seed)

        self.poisoned_test_dataset = PrePoisonedDataset(
            data_path=poison_data_path, prefix="test", transform=test_dataset.transform,
            poisoned_rate=1.0, seed=seed)