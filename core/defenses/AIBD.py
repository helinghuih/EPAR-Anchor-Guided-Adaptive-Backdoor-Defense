import copy
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from .base import Base
import core.defenses.analysis_utils as utils


class RelabeledDataset(Dataset):
    """Wraps a poisoned training set and overrides isolated sample labels with adversarial labels."""

    def __init__(self, base_dataset, new_labels, transform=None):
        self.base_dataset = base_dataset
        self.new_labels = new_labels
        self.transform = transform
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        if self.transform is not None:
            if isinstance(img, torch.Tensor):
                img = self.to_pil(img)
            img = self.transform(img)
        return img, int(self.new_labels[idx])


def aibd_pgd_batch(model, images, labels, device="cuda", eps=1.0, alpha=1.0 / 255.0, max_steps=255):
    """Dynamic active-subbatch L_inf-PGD probing matching AIBD's official pipeline."""
    model.eval()
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    bsz = images.size(0)

    with torch.no_grad():
        orig_preds = model(images).argmax(dim=1)

    counts = torch.zeros(bsz, dtype=torch.long, device=device)
    adv_labels = orig_preds.clone()

    active_idx = torch.arange(bsz, device=device)
    perturbed_active = images.clone().detach()
    orig_images_active = images
    orig_preds_active = orig_preds
    labels_active = labels
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for step in range(max_steps):
        perturbed_active.requires_grad_(True)
        outputs = model(perturbed_active)

        if step > 0:
            curr_preds = outputs.argmax(dim=1)
            flipped_mask = curr_preds != orig_preds_active
            if flipped_mask.any():
                adv_labels[active_idx[flipped_mask]] = curr_preds[flipped_mask]
                keep_mask = ~flipped_mask
                if not keep_mask.any():
                    active_idx = active_idx[:0]
                    break
                active_idx = active_idx[keep_mask]
                orig_images_active = orig_images_active[keep_mask]
                orig_preds_active = orig_preds_active[keep_mask]
                labels_active = labels_active[keep_mask]
                outputs = outputs[keep_mask]
        else:
            flipped_mask = None

        counts[active_idx] += 1
        model.zero_grad()
        loss = loss_fn(outputs, labels_active)
        grad_full = torch.autograd.grad(loss, perturbed_active)[0]

        if flipped_mask is not None and flipped_mask.any():
            keep_mask = ~flipped_mask
            grad_active = grad_full[keep_mask]
            perturbed_base = perturbed_active.detach()[keep_mask]
        else:
            grad_active = grad_full
            perturbed_base = perturbed_active.detach()

        with torch.no_grad():
            perturbed_next = perturbed_base + alpha * grad_active.sign()
            perturbed_next = torch.max(
                torch.min(perturbed_next, orig_images_active + eps), orig_images_active - eps
            )
            perturbed_active = torch.clamp(perturbed_next, 0.0, 1.0).detach()

    if active_idx.numel() > 0:
        with torch.no_grad():
            adv_labels[active_idx] = model(perturbed_active).argmax(dim=1)

    return counts.cpu().tolist(), adv_labels.cpu().tolist()


def select_aibd_isolated_indices(count_list, given_labels, target_label, ratio, scope="target"):
    total_n = len(count_list)
    k_select = max(1, int(total_n * ratio))
    if scope == "target":
        cand = [i for i, lbl in enumerate(given_labels) if int(lbl) == int(target_label)]
        if k_select > len(cand):
            return None
        cand.sort(key=lambda idx: count_list[idx], reverse=True)
        return cand[:k_select]
    return np.argsort(np.asarray(count_list))[::-1][:k_select].tolist()


def _aibd_step_lr(optimizer, epoch):
    lr = 0.1 if epoch < 5 else (0.01 if epoch < 10 else (0.001 if epoch < 15 else 0.0001))
    for group in optimizer.param_groups:
        group["lr"] = lr


class AIBD(Base):
    """Adversarial-Inspired Backdoor Defense (Yin et al., AAAI 2025) under EPAR's unified interface."""

    def __init__(self, model, poisoned_trainset, poisoned_testset, clean_testset,
                 num_classes, target_class, seed=666, deterministic=False):
        super().__init__(seed, deterministic)
        self.model = model
        self.poisoned_trainset = poisoned_trainset
        self.poisoned_testset = poisoned_testset
        self.clean_testset = clean_testset
        self.num_classes = num_classes
        self.target_class = target_class
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def probe_pgd_steps(self, cache_path=None, batch_size=512, eps=1.0, alpha=1.0 / 255.0, max_steps=255):
        if cache_path and osp.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            if isinstance(cached, dict) and len(cached.get("count_list", [])) == len(self.poisoned_trainset):
                return cached["count_list"], cached["adv_label_list"]

        self.model = self.model.to(self.device).eval()
        loader = DataLoader(self.poisoned_trainset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        count_list, adv_label_list = [], []
        for imgs, lbls in tqdm(loader, desc="AIBD PGD Probing", leave=False):
            c_b, adv_b = aibd_pgd_batch(
                self.model, imgs, lbls, device=self.device, eps=eps, alpha=alpha, max_steps=max_steps
            )
            count_list.extend(c_b)
            adv_label_list.extend(adv_b)

        if cache_path:
            os.makedirs(osp.dirname(osp.abspath(cache_path)), exist_ok=True)
            torch.save({"count_list": count_list, "adv_label_list": adv_label_list}, cache_path)
        return count_list, adv_label_list

    def _finetune_for_ratio(self, given_labels, adv_label_list, isolated_indices,
                            ori_acc, ori_asr, retrain_transform=None,
                            epochs=15, batch_size=128, drop_acc_threshold=3.5):
        new_labels = list(given_labels)
        for idx in isolated_indices:
            new_labels[idx] = int(adv_label_list[idx])

        ds = RelabeledDataset(self.poisoned_trainset, new_labels, transform=retrain_transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        net = copy.deepcopy(self.model).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        best_acc, best_asr, forgot = ori_acc, ori_asr, False
        temp_acc = ori_acc

        for ep in range(epochs):
            _aibd_step_lr(optimizer, ep)
            net.train()
            for imgs, targets in loader:
                imgs, targets = imgs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                criterion(net(imgs), targets).backward()
                optimizer.step()

            temp_acc, _ = utils._quick_eval(net, self.clean_testset, self.device, batch_size)
            temp_asr, _ = utils._quick_eval(net, self.poisoned_testset, self.device, batch_size)

            if ep == 5 and temp_acc < ori_acc - drop_acc_threshold * 2.0:
                forgot = True
                break
            best_acc, best_asr = temp_acc, temp_asr

        accepted = (not forgot) and (temp_acc >= ori_acc - drop_acc_threshold)
        return net, best_acc, best_asr, accepted, forgot

    def train(self, schedule):
        self.model = self.model.to(self.device).eval()
        given_labels = schedule["given_labels"]
        poisoned_set = schedule["poisoned_set"]
        total_n = len(self.poisoned_trainset)

        ori_acc, _ = utils._quick_eval(self.model, self.clean_testset, self.device, 128)
        ori_asr, _ = utils._quick_eval(self.model, self.poisoned_testset, self.device, 128)

        count_list, adv_label_list = self.probe_pgd_steps(
            cache_path=schedule.get("cache_path"),
            batch_size=schedule.get("pgd_bs", 512),
            eps=schedule.get("adv_epsilon", 1.0),
            alpha=schedule.get("adv_alpha", 1.0 / 255.0),
            max_steps=schedule.get("adv_steps", 255),
        )

        if schedule.get("target_mode", "oracle") == "oracle":
            guess_label = self.target_class
        else:
            top_k = max(50, int(total_n * schedule.get("label_rate", 0.001)))
            top_idx = np.argsort(np.asarray(count_list))[::-1][:top_k]
            votes = [0] * self.num_classes
            for idx in top_idx:
                votes[int(given_labels[idx])] += 1
            guess_label = int(np.argmax(votes))

        q_list = schedule["q_list"]
        scope = schedule.get("scope", "target")
        q_strategy = schedule.get("q_strategy", "progressive")

        chosen_q, chosen_iso = 0.0, []
        final_acc, final_asr = ori_acc, ori_asr

        for q in q_list:
            iso = select_aibd_isolated_indices(count_list, given_labels, guess_label, ratio=q, scope=scope)
            if iso is None:
                continue
            net_q, acc_q, asr_q, accepted, forgot = self._finetune_for_ratio(
                given_labels, adv_label_list, iso, ori_acc, ori_asr,
                retrain_transform=schedule.get("retrain_transform"),
                epochs=schedule.get("retrain_epochs", 15),
                batch_size=schedule.get("batch_size", 128),
                drop_acc_threshold=schedule.get("drop_acc_threshold", 3.5),
            )
            chosen_q, chosen_iso, final_acc, final_asr = q, iso, acc_q, asr_q
            status = "Forgot" if forgot else ("Accepted" if accepted else "Rollback")
            print(f"  [AIBD q={q*100:4.1f}%] ACC={acc_q:6.2f}% ASR={asr_q:6.2f}% ({status})")
            if q_strategy == "fixed" or accepted:
                self.model = net_q
                break

        iso_set = set(chosen_iso)
        tp = len(iso_set & poisoned_set)
        fp = len(iso_set - poisoned_set)
        tpr = (tp / len(poisoned_set) * 100.0) if poisoned_set else 0.0
        fpr = (fp / (total_n - len(poisoned_set)) * 100.0) if (total_n - len(poisoned_set)) > 0 else 0.0
        return {
            "q": chosen_q,
            "tp": tp,
            "fp": fp,
            "tpr": tpr,
            "fpr": fpr,
            "acc": final_acc,
            "asr": final_asr,
        }
