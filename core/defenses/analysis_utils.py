import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
import copy
import random
from ..utils.accuracy import accuracy

class SCELoss(torch.nn.Module):
    def __init__(self, num_classes, alpha=0.1, beta=1.0, reduction='mean'):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        if self.reduction == 'mean':
            rce = rce.mean()
        loss = self.alpha * ce + self.beta * rce
        return loss


def fft_analysis(image, high_freq_threshold_ratio=0.25):
    """Calculate High Frequency Energy"""
    if torch.is_tensor(image):
        img = image.clone().detach()
    else:
        img = torch.tensor(image)
    if len(img.shape) == 4: img = img.squeeze(0)
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


def analyze_batch_with_ddn(model, data_input, device, num_classes, ddn_steps=100):
    model.eval()
    all_results = []
    sce_loss_fn = SCELoss(num_classes=num_classes, reduction='none')

    if hasattr(data_input, '__iter__') and hasattr(data_input, 'dataset'):
        for batch_data in tqdm(data_input, desc="DDN Analysis"):
            if len(batch_data) == 2:
                batch_images, batch_labels = batch_data
            elif len(batch_data) == 3:
                batch_images, batch_labels, _ = batch_data

            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            batch_results = _process_batch(model, batch_images, batch_labels, device, ddn_steps, sce_loss_fn)
            all_results.extend(batch_results)
    else:
        raise ValueError("Input must be a DataLoader")
    return all_results


def _process_batch(model, batch_images, batch_labels, device, ddn_steps, sce_loss_fn):
    batch_results = []
    current_batch_size = batch_images.size(0)
    for i in range(current_batch_size):
        image = batch_images[i:i + 1]
        label = batch_labels[i:i + 1]

        with torch.no_grad():
            orig_logits = model(image)
            orig_pred = orig_logits.argmax(dim=1)
            pre_attack_sce = sce_loss_fn(orig_logits, label).item()

        pre_attack_high_freq_energy = fft_analysis(image.squeeze(0))

        # DDN Attack Logic
        delta = torch.zeros_like(image, requires_grad=True)
        optimizer = torch.optim.SGD([delta], lr=1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ddn_steps, eta_min=0.01)
        norm = torch.full((1,), 1.0, device=device, dtype=torch.float)
        worst_norm = torch.max(image, 1 - image).view(1, -1).norm(p=2, dim=1)

        for step in range(ddn_steps):
            scheduler.step()
            adv_image = image + delta
            logits = model(adv_image)
            pred_labels = logits.argmax(1)
            if pred_labels.item() != orig_pred.item(): break
            loss = -torch.nn.functional.cross_entropy(logits, orig_pred, reduction='sum')
            optimizer.zero_grad()
            loss.backward()
            grad_norms = delta.grad.view(1, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            if (grad_norms == 0).any(): delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer.step()
            norm.mul_(1 - (2 * (pred_labels != orig_pred).float() - 1) * 0.05)
            norm = torch.min(norm, worst_norm)
            delta.data.mul_((norm / delta.data.view(1, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(image).clamp_(0, 1).sub_(image)

        adv_image = image + delta
        post_attack_high_freq_energy = fft_analysis(adv_image.squeeze(0))
        batch_results.append({
            'freq_diff_pre_post_energy': post_attack_high_freq_energy - pre_attack_high_freq_energy,
            'pre_attack_sce': pre_attack_sce,
            'pre_attack_hfe': pre_attack_high_freq_energy,
            'post_attack_hfe': post_attack_high_freq_energy,
        })
    return batch_results


def pgd_attack_steps_batch(model, images, labels, device='cuda', eps=8 / 255, alpha=1 / 255, max_steps=50):
    """Batch PGD Step Calculation"""
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    batch_size = images.size(0)

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


class CleanSubsetDataset(Dataset):
    def __init__(self, samples_indices, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.indices = samples_indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        _, idx = self.indices[i]
        # Handle original GTSRB item
        img, label = self.original_dataset[idx]
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        return img, int(label)


class MatchSampleDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        image, label = self.original_dataset[idx]
        return image, label, idx


def train_cft_model(model_init, dataset, p_test, clean_test, schedule, log, device='cuda'):
    """
    Confusion Fine-Tuning (CFT):
    Train on the dataset with SHUFFLED labels to destroy clean features.
    """
    log(f"\n--- [Phase 1] Training Exposure Model (CFT) ---")

    loader = DataLoader(dataset, batch_size=schedule['batch_size'], shuffle=True, num_workers=schedule['num_workers'])
    model = copy.deepcopy(model_init).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=schedule['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedule['schedule'], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(schedule['epochs']):
        total_loss = 0
        model.train()
        for img, label in loader:
            img = img.to(device)

            # [CFT 核心] 打乱标签
            perm = torch.randperm(label.size(0))
            shuffled_label = label[perm].to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, shuffled_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == schedule['epochs']:
            acc_clean, _ = _quick_eval(model, clean_test, device, schedule['batch_size'])
            acc_pois, _ = _quick_eval(model, p_test, device, schedule['batch_size'])
            log(f"CFT Epoch {epoch + 1}/{schedule['epochs']} | "
                f"Loss: {total_loss / len(loader):.4f} | "
                f"ASR: {acc_pois:.2f}% | Acc: {acc_clean:.2f}%")

    return model


def train_clean_model(model_init, dataset, p_test, clean_test, schedule, stage_name, log, device='cuda'):
    log(f"\n--- [Phase 3] Starting {stage_name} ---")
    loader = DataLoader(dataset, batch_size=schedule['batch_size'], shuffle=True, num_workers=schedule['num_workers'],
                        drop_last=True)
    model = copy.deepcopy(model_init).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=schedule['lr'], momentum=0.9, weight_decay=5e-4)
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
    outputs_list = []
    labels_list = []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            out = model(img)
            outputs_list.append(out.cpu())
            labels_list.append(label.cpu())
    outputs = torch.cat(outputs_list)
    labels = torch.cat(labels_list)
    prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
    return prec1.item(), prec5.item()

def get_few_shot_clean_subset(dataset, num_classes, shots, seed=666):
    print(f"Selecting {shots} CLEAN samples per class (Total classes: {num_classes})...")

    class_indices = {i: [] for i in range(num_classes)}
    selected_indices = []

    all_indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(all_indices)

    finished_classes = 0
    poisoned_indices = set()
    if hasattr(dataset, 'poisoned_set') and dataset.poisoned_set is not None:
        poisoned_indices = dataset.poisoned_set
        print(f"  [Info] Found poisoned_set with {len(poisoned_indices)} indices. These will be skipped.")

    all_labels = None
    if hasattr(dataset, 'targets'):
        all_labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        all_labels = dataset.labels
    elif hasattr(dataset, '_samples'):
        all_labels = [x[1] for x in dataset._samples]

    if all_labels is not None:
        for idx in all_indices:
            if idx in poisoned_indices:
                continue

            label = int(all_labels[idx])

            if len(class_indices[label]) < shots:
                class_indices[label].append(idx)
                selected_indices.append(idx)

                if len(class_indices[label]) == shots:
                    finished_classes += 1

            if finished_classes >= num_classes:
                break
    else:
        print("[Warning] Dataset labels not found directly. Iterating (slow)...")
        for idx in all_indices:
            if idx in poisoned_indices:
                continue

            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            label = int(label)

            if len(class_indices[label]) < shots:
                class_indices[label].append(idx)
                selected_indices.append(idx)
                if len(class_indices[label]) == shots:
                    finished_classes += 1
            if finished_classes >= num_classes:
                break

    print(f"Selected total {len(selected_indices)} GUARANTEED CLEAN samples for guidance.")
    return Subset(dataset, selected_indices)


def get_clean_samples_of_target(dataset, target_class, count=30):
    print(f"Searching for {count} CLEAN samples of class {target_class}...")
    clean_indices = []
    poisoned_indices = set()
    if hasattr(dataset, 'poisoned_set') and dataset.poisoned_set is not None:
        poisoned_indices = dataset.poisoned_set

    all_labels = None
    if hasattr(dataset, 'targets'):
        all_labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        all_labels = dataset.labels
    elif hasattr(dataset, '_samples'):
        all_labels = [x[1] for x in dataset._samples]
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        if idx in poisoned_indices: continue
        if all_labels:
            label = int(all_labels[idx])
        else:
            _, label = dataset[idx]
            label = int(label if not isinstance(label, torch.Tensor) else label.item())

        if label == target_class:
            clean_indices.append(idx)
            if len(clean_indices) >= count:
                break

    print(f"Found {len(clean_indices)} verified clean samples for Anchor.")
    return clean_indices


def train_guided_model(model, subset, device='cuda', epochs=20):
    loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=4)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    print(f"Training guided model for {epochs} epochs...")
    for epoch in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return model


def calculate_guided_sce_values(dataset, model_factory, num_classes, shots, device='cuda', batch_size=128,
                                num_workers=4):

    model = model_factory()

    clean_subset = get_few_shot_clean_subset(
        dataset,
        num_classes=num_classes,
        shots=shots
    )

    model = train_guided_model(model, clean_subset, device=device, epochs=20)

    model.eval()
    criterion = SCELoss(num_classes=num_classes, reduction='none')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sce_values = []
    with torch.no_grad():
        for img, target in tqdm(loader, desc="Calculating Guided SCE"):
            img, target = img.to(device), target.to(device)
            output = model(img)
            loss = criterion(output, target)
            sce_values.append(loss.cpu().numpy())

    return np.concatenate(sce_values)
