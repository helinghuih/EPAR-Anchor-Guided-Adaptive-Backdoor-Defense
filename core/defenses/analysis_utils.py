import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToPILImage, ToTensor
from tqdm import tqdm
from ..utils.accuracy import accuracy


def fft_analysis(image, high_freq_threshold_ratio=0.25):
    if torch.is_tensor(image):
        img = image.clone().detach()
    else:
        img = torch.tensor(image)
    if len(img.shape) == 4:
        img = img.squeeze(0)
    if len(img.shape) == 3:
        if img.shape[0] == 3:
            img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        elif img.shape[0] == 1:
            img = img.squeeze(0)
    img_np = img.cpu().numpy().astype(np.float32)
    fft_result = np.fft.fft2(img_np)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    h, w = img_np.shape
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    freq_threshold = min(h, w) * high_freq_threshold_ratio
    high_freq_mask = distances > freq_threshold
    magnitude_squared = magnitude ** 2
    high_freq_energy = np.sum(magnitude_squared[high_freq_mask])
    return float(high_freq_energy)


def analyze_batch_with_ddn(model, data_input, device, num_classes=10, ddn_steps=100):
    model.eval()
    all_results = []

    if hasattr(data_input, '__iter__') and hasattr(data_input, 'dataset'):
        for batch_data in tqdm(data_input, desc="DDN Probing", leave=False):
            batch_images = batch_data[0].to(device)
            batch_results = _process_batch(model, batch_images, device, ddn_steps)
            all_results.extend(batch_results)
    else:
        raise ValueError("Input must be a DataLoader")
    return all_results


def _process_batch(model, batch_images, device, ddn_steps, init_norm=1.0, gamma=0.05):
    model.eval()
    B = batch_images.size(0)
    images = batch_images.clone().detach().to(device)

    with torch.no_grad():
        orig_preds = model(images).argmax(dim=1)

    delta = torch.zeros_like(images, device=device)
    norm = torch.full((B,), init_norm, device=device, dtype=torch.float)
    worst_norm = torch.max(images, 1.0 - images).view(B, -1).norm(p=2, dim=1)

    best_l2 = worst_norm.clone()
    best_adv = images.clone()
    adv_found = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(ddn_steps):
        lr = 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * step / max(ddn_steps, 1)))
        adv_images = (images + delta).detach().requires_grad_(True)
        logits = model(adv_images)
        pred_labels = logits.argmax(dim=1)

        is_adv = (pred_labels != orig_preds)
        curr_l2 = (adv_images.detach() - images).view(B, -1).norm(p=2, dim=1)
        improved = is_adv & (curr_l2 < best_l2)
        if improved.any():
            best_l2[improved] = curr_l2[improved]
            best_adv[improved] = adv_images.detach()[improved]
            adv_found[improved] = True

        loss = F.cross_entropy(logits, orig_preds, reduction='sum')
        grad = torch.autograd.grad(loss, adv_images)[0]
        grad_norms = grad.view(B, -1).norm(p=2, dim=1)
        zero_mask = (grad_norms < 1e-12)
        if zero_mask.any():
            grad[zero_mask] = torch.randn_like(grad[zero_mask])
            grad_norms = grad.view(B, -1).norm(p=2, dim=1)
        scaled_grad = grad / grad_norms.view(B, 1, 1, 1)

        delta = delta + lr * scaled_grad
        factor = torch.where(is_adv, 1.0 - gamma, 1.0 + gamma)
        norm = torch.clamp(norm * factor, min=1e-4)
        norm = torch.min(norm, worst_norm)

        delta_norms = delta.view(B, -1).norm(p=2, dim=1).clamp(min=1e-12)
        delta = delta * (norm / delta_norms).view(B, 1, 1, 1)
        delta = (images + delta).clamp(0.0, 1.0) - images

    final_adv = torch.where(adv_found.view(B, 1, 1, 1), best_adv, (images + delta).clamp(0.0, 1.0))

    batch_results = []
    for i in range(B):
        pre_hfe = fft_analysis(images[i])
        post_hfe = fft_analysis(final_adv[i])
        batch_results.append({
            'freq_diff_pre_post_energy': post_hfe - pre_hfe,
            'pre_attack_hfe': pre_hfe,
            'post_attack_hfe': post_hfe,
            'min_l2_norm': float(best_l2[i].item()),
        })
    return batch_results


def pgd_attack_steps_batch(model, images, labels, device='cuda', eps=None, alpha=0.5 / 255, max_steps=50):
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    batch_size = images.size(0)
    if eps is None:
        eps = max_steps * alpha

    step_counts = torch.full((batch_size,), max_steps, device=device, dtype=torch.long)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)
        already_wrong = (preds != labels)
        step_counts[already_wrong] = 0
        active_mask[already_wrong] = False

    if not active_mask.any():
        return step_counts.cpu().tolist()

    adv_images = images.clone().detach()

    for step in range(1, max_steps + 1):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        cost = loss.sum()
        grad = torch.autograd.grad(cost, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        with torch.no_grad():
            curr_preds = model(adv_images).argmax(dim=1)
            is_wrong = (curr_preds != labels)
            just_flipped = is_wrong & active_mask
            step_counts[just_flipped] = step
            active_mask[just_flipped] = False

        if not active_mask.any():
            break

    return step_counts.cpu().tolist()


class RetrainTransformDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        if transform is None:
            transform = Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor()
            ])
        self.transform = transform
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.subset_dataset)

    def __getitem__(self, i):
        img, label = self.subset_dataset[i]
        if self.transform is not None:
            if isinstance(img, torch.Tensor):
                img = self.to_pil(img)
            img = self.transform(img)
        if isinstance(label, torch.Tensor):
            label = label.item()
        return img, int(label)


class MatchSampleDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        image, label = self.original_dataset[idx]
        return image, label, idx


def train_clean_model(model_init, dataset, p_test, clean_test, schedule, stage_name, log, device='cuda'):
    retrain_transform = schedule.get('transform', None)
    if stage_name == "Retraining" and schedule.get('use_retrain_aug', True) and retrain_transform is not False:
        dataset = RetrainTransformDataset(dataset, transform=retrain_transform)

    loader = DataLoader(dataset, batch_size=schedule['batch_size'], shuffle=True, num_workers=schedule['num_workers'],
                        drop_last=True)
    model = copy.deepcopy(model_init).to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=schedule['lr'],
        momentum=schedule.get('momentum', 0.9),
        weight_decay=schedule.get('weight_decay', 5e-4)
    )
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=schedule['schedule'], gamma=schedule['gamma'])
    crit = nn.CrossEntropyLoss()

    for epoch in range(schedule['epochs']):
        model.train()
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            opt.zero_grad()
            loss = crit(model(img), label)
            loss.backward()
            opt.step()
        sched.step()

        if (epoch + 1) % schedule.get('test_epoch_interval', 5) == 0:
            acc_clean, _ = _quick_eval(model, clean_test, device, schedule['batch_size'])
            acc_pois, _ = _quick_eval(model, p_test, device, schedule['batch_size'])
            log(f"[{stage_name}] Epoch:{epoch + 1}/{schedule['epochs']} | ASR: {acc_pois:.2f}% | Acc: {acc_clean:.2f}%")
    return model


def _quick_eval(model, dataset, device, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    outputs_list, labels_list = [], []
    with torch.no_grad():
        for img, label in loader:
            outputs_list.append(model(img.to(device)).cpu())
            labels_list.append(label.cpu())
    outputs = torch.cat(outputs_list)
    labels = torch.cat(labels_list)
    prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
    return prec1.item(), prec5.item()


def get_few_shot_clean_subset(dataset, num_classes, shots, seed=666):
    class_indices = {i: [] for i in range(num_classes)}
    selected_indices = []

    all_indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(all_indices)

    finished_classes = 0
    poisoned_indices = set()
    if hasattr(dataset, 'poisoned_set') and dataset.poisoned_set is not None:
        poisoned_indices = dataset.poisoned_set

    all_labels = None
    if hasattr(dataset, 'targets'):
        all_labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        all_labels = dataset.labels
    elif hasattr(dataset, 'samples'):
        all_labels = [x[1] for x in dataset.samples]
    elif hasattr(dataset, '_samples'):
        all_labels = [x[1] for x in dataset._samples]

    for idx in all_indices:
        if idx in poisoned_indices:
            continue
        if all_labels is not None:
            label = int(all_labels[idx])
        else:
            _, lbl = dataset[idx]
            label = int(lbl.item() if isinstance(lbl, torch.Tensor) else lbl)

        if len(class_indices[label]) < shots:
            class_indices[label].append(idx)
            selected_indices.append(idx)
            if len(class_indices[label]) == shots:
                finished_classes += 1

        if finished_classes >= num_classes:
            break

    return Subset(dataset, selected_indices)