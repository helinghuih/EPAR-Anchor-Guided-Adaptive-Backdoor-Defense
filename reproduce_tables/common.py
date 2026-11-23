#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared dataset, attack, model, probing, and metric helpers for reproduce_tables/."""

import os
import os.path as osp
import random
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core
import core.defenses.analysis_utils as utils
from core.defenses.epar import kmeans_pytorch
from core.utils.tiny_imagenet_utils import build_tiny_imagenet_views


def build_model(arch, num_classes):
    arch_key = arch.lower()
    if arch_key == "resnet18":
        return core.models.ResNet(18, num_classes=num_classes)
    if arch_key == "vgg19_bn":
        return core.models.vgg19_bn(num_classes=num_classes)
    if arch_key == "mobilenetv2":
        return torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
    if arch_key == "densenet121":
        return torchvision.models.densenet121(weights=None, num_classes=num_classes)
    raise ValueError(f"Unsupported architecture: {arch}. Choose from resnet18, vgg19_bn, mobilenetv2, densenet121.")


def load_checkpoint(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd.get("model_state_dict", sd))
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()})
    return model


def gen_wanet_grids(img_size=64, k=4, s=0.5, seed=666):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.interpolate(ins, size=img_size, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=img_size)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    torch.set_rng_state(rng_state)
    return identity_grid, noise_grid * s


def get_dataset_config(dataset_name):
    if dataset_name == "CIFAR10":
        return {
            "num_classes": 10,
            "target_class": 0,
            "shots": 30,
            "img_size": 32,
            "cul_lr": 1e-4,
            "default_retrain_aug": True,
            "default_drop_acc_thr": 3.5,
            "retrain_schedule": {
                "batch_size": 128, "num_workers": 4, "epochs": 100,
                "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
                "gamma": 0.1, "schedule": [75, 90], "test_epoch_interval": 10,
                "use_retrain_aug": True,
                "transform": Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor()]),
            },
        }
    if dataset_name == "GTSRB":
        return {
            "num_classes": 43,
            "target_class": 1,
            "shots": 30,
            "img_size": 32,
            "cul_lr": 1e-5,
            "default_retrain_aug": False,
            "default_drop_acc_thr": 6.5,
            "retrain_schedule": {
                "batch_size": 128, "num_workers": 4, "epochs": 100,
                "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
                "gamma": 0.1, "schedule": [30, 60], "test_epoch_interval": 10,
                "use_retrain_aug": False, "transform": False,
            },
        }
    if dataset_name == "TinyImageNet":
        return {
            "num_classes": 200,
            "target_class": 0,
            "shots": 10,
            "img_size": 64,
            "cul_lr": 1e-5,
            "default_retrain_aug": False,
            "default_drop_acc_thr": 3.5,
            "retrain_schedule": {
                "batch_size": 128, "num_workers": 8, "epochs": 100,
                "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
                "gamma": 0.1, "schedule": [75, 90], "test_epoch_interval": 10,
                "use_retrain_aug": False, "transform": False,
            },
        }
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_base_datasets(dataset_name, data_dir="./data"):
    if dataset_name == "CIFAR10":
        tf = Compose([ToTensor()])
        return (
            torchvision.datasets.CIFAR10(data_dir, train=True, transform=tf, download=True),
            torchvision.datasets.CIFAR10(data_dir, train=False, transform=tf, download=True),
        )
    if dataset_name == "GTSRB":
        tf = Compose([Resize((32, 32)), ToTensor()])
        return (
            torchvision.datasets.GTSRB(data_dir, split="train", transform=tf, download=True),
            torchvision.datasets.GTSRB(data_dir, split="test", transform=tf, download=True),
        )
    if dataset_name == "TinyImageNet":
        views = build_tiny_imagenet_views(root=osp.join(data_dir, "tiny-imagenet-200"))
        return views["train"], views["validation"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_attack(dataset_name, attack_name, trainset, testset, ckpt_dir, seed=666, arch="resnet18"):
    cfg = get_dataset_config(dataset_name)
    num_classes, target_class, img_size = cfg["num_classes"], cfg["target_class"], cfg["img_size"]
    t_idx = 1 if dataset_name == "GTSRB" else 0
    det_flag = dataset_name == "CIFAR10"
    model = build_model(arch, num_classes)
    loss_fn = nn.CrossEntropyLoss()

    if attack_name == "BadNets":
        patch = 6 if dataset_name == "TinyImageNet" else 3
        ch = 1 if dataset_name == "CIFAR10" else 3
        pattern = torch.zeros((ch, img_size, img_size), dtype=torch.uint8)
        pattern[:, -patch:, -patch:] = 255
        weight = torch.zeros((ch, img_size, img_size), dtype=torch.float32)
        weight[:, -patch:, -patch:] = 1.0
        return core.BadNets(
            train_dataset=trainset, test_dataset=testset, model=model, loss=loss_fn,
            y_target=target_class, poisoned_rate=0.1, pattern=pattern, weight=weight,
            poisoned_transform_train_index=t_idx, poisoned_transform_test_index=t_idx,
            poisoned_target_transform_index=0, seed=seed, deterministic=det_flag,
        )
    if attack_name == "Blended":
        img = Image.open(osp.join(PROJECT_ROOT, "resource/blended/hello_kitty.jpeg")).convert("RGB").resize((img_size, img_size))
        pattern = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).to(torch.uint8)
        weight = torch.full((3, img_size, img_size), 0.2, dtype=torch.float32)
        return core.Blended(
            train_dataset=trainset, test_dataset=testset, model=model, loss=loss_fn,
            pattern=pattern, weight=weight, y_target=target_class, poisoned_rate=0.1,
            poisoned_transform_train_index=t_idx, poisoned_transform_test_index=t_idx,
            poisoned_target_transform_index=0, seed=seed, deterministic=det_flag,
        )
    if attack_name == "Refool":
        ref_dir = osp.join(PROJECT_ROOT, "resource/refool/Refool-SelectedReflectionImages/selected_out-images")
        ref_cache = osp.join(PROJECT_ROOT, "experiments", f"{dataset_name}_Refool_cache_seed{seed}")
        os.makedirs(ref_cache, exist_ok=True)
        return core.Refool(
            train_dataset=trainset, test_dataset=testset, model=model, loss=loss_fn,
            y_target=target_class, poisoned_rate=0.1, reflection_dir=ref_dir,
            save_dir=ref_cache, seed=seed, deterministic=det_flag,
        )
    if attack_name == "WaNet":
        ig_p, ng_p = osp.join(ckpt_dir, "identity_grid.pth"), osp.join(ckpt_dir, "noise_grid.pth")
        if not osp.exists(ig_p):
            ig_p, ng_p = osp.join(ckpt_dir, "identity_grid.pt"), osp.join(ckpt_dir, "noise_grid.pt")
        if osp.exists(ig_p) and osp.exists(ng_p):
            ig, ng = torch.load(ig_p, map_location="cpu"), torch.load(ng_p, map_location="cpu")
        else:
            ig, ng = gen_wanet_grids(img_size=img_size, k=4, s=0.5, seed=seed)
        return core.WaNet(
            train_dataset=trainset, test_dataset=testset, model=model, loss=loss_fn,
            y_target=target_class, poisoned_rate=0.1, identity_grid=ig, noise_grid=ng,
            noise=False, poisoned_transform_train_index=t_idx, poisoned_transform_test_index=t_idx,
            poisoned_target_transform_index=0, seed=seed, deterministic=det_flag,
        )
    if attack_name == "SIG":
        return core.SIG(
            train_dataset=trainset, test_dataset=testset, model=model, loss=loss_fn,
            y_target=target_class, poisoned_rate=0.05, delta=40 / 255.0, f=6,
            poisoned_transform_train_index=t_idx, poisoned_transform_test_index=t_idx,
            poisoned_target_transform_index=0, seed=seed, deterministic=det_flag,
        )
    raise ValueError(f"Unsupported attack: {attack_name}")


def get_clean_targets(trainset):
    for attr in ("targets", "labels"):
        if hasattr(trainset, attr):
            return [int(x) for x in getattr(trainset, attr)]
    for attr in ("samples", "_samples"):
        if hasattr(trainset, attr):
            return [int(x[1]) for x in getattr(trainset, attr)]
    return [int(trainset[i][1]) for i in range(len(trainset))]


def compute_tpr_fpr(kept_indices, total_n, poisoned_set):
    discarded = set(range(total_n)) - set(kept_indices)
    tp = len(discarded & poisoned_set)
    fp = len(discarded - poisoned_set)
    total_p = len(poisoned_set)
    total_c = total_n - total_p
    return tp, fp, (tp / total_p * 100.0 if total_p else 0.0), (fp / total_c * 100.0 if total_c else 0.0)


def extract_epar_probes(dataset_name, attack_name, p_train, orig_targets, model, cache_dir, seed, device):
    """Extracts Stage-2 penultimate features, cached 100-step DDN DeltaE, and 50-step PGD steps."""
    cfg = get_dataset_config(dataset_name)
    num_classes, target_class, shots = cfg["num_classes"], cfg["target_class"], cfg["shots"]
    poisoned_set = set(int(i) for i in p_train.poisoned_set)
    total_n = len(p_train)

    given_labels = [
        target_class if (i in poisoned_set or int(orig_targets[i]) == target_class) else int(orig_targets[i])
        for i in range(total_n)
    ]
    g2_indices = [i for i, lbl in enumerate(given_labels) if lbl == target_class]
    non_g2_indices = [i for i, lbl in enumerate(given_labels) if lbl != target_class]

    trusted_clean_set = utils.get_few_shot_clean_subset(p_train, num_classes=num_classes, shots=shots, seed=seed)
    trusted_indices = [int(i) for i in trusted_clean_set.indices]
    anchor_pos = [i for i in range(len(trusted_clean_set)) if int(trusted_clean_set[i][1]) == target_class]
    anchor_train_indices = [int(trusted_clean_set.indices[i]) for i in anchor_pos]
    anchor_loader = DataLoader(Subset(trusted_clean_set, anchor_pos), batch_size=32, shuffle=False)

    model = model.to(device).eval()
    feat_buf = {}
    last_linear = next(m for _, m in reversed(list(model.named_modules())) if isinstance(m, nn.Linear))
    handle = last_linear.register_forward_hook(lambda m, inp, out: feat_buf.__setitem__("f", inp[0].detach().flatten(1)))

    anchor_feats = []
    with torch.no_grad():
        for imgs, _ in anchor_loader:
            model(imgs.to(device))
            anchor_feats.append(feat_buf["f"].cpu())
    anchor_center = F.normalize(F.normalize(torch.cat(anchor_feats, dim=0), p=2, dim=1).mean(dim=0, keepdim=True), p=2, dim=1)

    g2_feats, g2_map = [], []
    with torch.no_grad():
        for imgs, _, idxs in DataLoader(utils.MatchSampleDataset(p_train, g2_indices), batch_size=128, shuffle=False):
            model(imgs.to(device))
            g2_feats.append(feat_buf["f"].cpu())
            g2_map.extend(idxs.tolist())
    handle.remove()
    g2_feats_norm = F.normalize(torch.cat(g2_feats, dim=0), p=2, dim=1)

    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{dataset_name}_{attack_name}_seed{seed}"
    ddn_path = osp.join(cache_dir, f"{tag}_ddn_results_v2.pth")
    pgd_path = osp.join(cache_dir, f"{tag}_pgd_g2_results_v2.pth")

    hfe_map = torch.load(ddn_path) if osp.exists(ddn_path) else {}
    missing_ddn = [i for i in g2_indices if i not in hfe_map]
    if missing_ddn:
        res = utils.analyze_batch_with_ddn(
            model, DataLoader(utils.MatchSampleDataset(p_train, missing_ddn), batch_size=64, shuffle=False),
            device, num_classes=num_classes, ddn_steps=100,
        )
        for i, r in enumerate(res):
            hfe_map[missing_ddn[i]] = r["freq_diff_pre_post_energy"]
        torch.save(hfe_map, ddn_path)

    pgd_map = torch.load(pgd_path) if osp.exists(pgd_path) else {}
    missing_pgd = [i for i in g2_indices if i not in pgd_map]
    if missing_pgd:
        for img, lbl, idxs in DataLoader(utils.MatchSampleDataset(p_train, missing_pgd), batch_size=64, shuffle=False):
            steps = utils.pgd_attack_steps_batch(model, img, lbl, device=device)
            for i, oid in enumerate(idxs):
                pgd_map[oid.item()] = steps[i]
        torch.save(pgd_map, pgd_path)

    def get_hfe(idx):
        v = hfe_map[idx]
        return v["freq_diff_pre_post_energy"] if isinstance(v, dict) else float(v)

    def agg_hfe(indices):
        if not indices:
            return 0.0
        vals = [get_hfe(i) for i in indices]
        return float(np.median(vals)) if shots <= 10 else float(np.mean(vals))

    return {
        "total_n": total_n,
        "poisoned_set": poisoned_set,
        "g2_indices": g2_indices,
        "non_g2_indices": non_g2_indices,
        "trusted_indices": trusted_indices,
        "anchor_center": anchor_center,
        "g2_feats_norm": g2_feats_norm,
        "g2_map": g2_map,
        "get_hfe": get_hfe,
        "agg_hfe": agg_hfe,
        "hfe_anchor": agg_hfe([i for i in anchor_train_indices if i in hfe_map]),
        "pgd_map": pgd_map,
    }


def cluster_and_split(probes, seed, device, init_anchor=True):
    torch.manual_seed(seed)
    labels, _ = kmeans_pytorch(
        probes["g2_feats_norm"], num_clusters=2, device=device,
        init_anchor=(probes["anchor_center"] if init_anchor else None),
    )
    m0, m1 = (labels == 0).cpu(), (labels == 1).cpu()
    c0 = F.normalize(probes["g2_feats_norm"][m0].mean(dim=0, keepdim=True), p=2, dim=1)
    c1 = F.normalize(probes["g2_feats_norm"][m1].mean(dim=0, keepdim=True), p=2, dim=1)
    d_cos0 = float(1.0 - F.cosine_similarity(probes["anchor_center"], c0).item())
    d_cos1 = float(1.0 - F.cosine_similarity(probes["anchor_center"], c1).item())
    idx0 = [probes["g2_map"][i] for i in m0.nonzero(as_tuple=True)[0].tolist()]
    idx1 = [probes["g2_map"][i] for i in m1.nonzero(as_tuple=True)[0].tolist()]
    hfe0, hfe1 = probes["agg_hfe"](idx0), probes["agg_hfe"](idx1)
    return (idx0, hfe0, d_cos0), (idx1, hfe1, d_cos1)


def stage3_prune(cand_indices, probes, bias, s3_mode="combined", gamma=0.40):
    h_arr = np.asarray([probes["get_hfe"](i) for i in cand_indices], dtype=np.float64)
    p_arr = np.asarray([probes["pgd_map"][i] for i in cand_indices], dtype=np.float64)

    lo_h, hi_h = np.percentile(h_arr, 2), np.percentile(h_arr, 98)
    if hi_h - lo_h < 1e-6:
        lo_h, hi_h = h_arr.min(), h_arr.max()
    norm_h = np.clip((h_arr - lo_h) / (hi_h - lo_h + 1e-6), 0.0, 1.0)
    clean_h = norm_h if bias < 0 else (1.0 - norm_h)

    lo_p, hi_p = np.percentile(p_arr, 2), np.percentile(p_arr, 98)
    if hi_p - lo_p < 1e-6:
        lo_p, hi_p = p_arr.min(), p_arr.max()
    norm_p = np.clip((p_arr - lo_p) / (hi_p - lo_p + 1e-6), 1e-3, 1.0)

    scores = norm_p if s3_mode == "pgd_only" else (clean_h if s3_mode == "freq_only" else clean_h * norm_p)
    order = np.argsort(-scores)
    keep_cnt = max(1, int(len(cand_indices) * (1.0 - gamma)))
    g2_ref = [cand_indices[i] for i in order[:keep_cnt]]
    kept = list(dict.fromkeys(probes["non_g2_indices"] + g2_ref + probes["trusted_indices"]))
    tp, fp, tpr, fpr = compute_tpr_fpr(kept, probes["total_n"], probes["poisoned_set"])
    return tp, fp, tpr, fpr, kept


def retrain_and_eval(dataset_name, p_train, p_test, testset, kept_indices, seed, device, arch="resnet18"):
    cfg = get_dataset_config(dataset_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(arch, cfg["num_classes"]).to(device)
    trained = utils.train_clean_model(
        model, Subset(p_train, kept_indices), p_test, testset, cfg["retrain_schedule"], "Retraining", print, device
    )
    acc, _ = utils._quick_eval(trained, testset, device, cfg["retrain_schedule"]["batch_size"])
    asr, _ = utils._quick_eval(trained, p_test, device, cfg["retrain_schedule"]["batch_size"])
    return acc, asr
