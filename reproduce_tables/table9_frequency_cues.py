#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 9 Reproduction: Controlled Comparison of Different Frequency Cues on CIFAR-10.

Compares four frequency cues under the same target subset, anchor-initialized K-Means
clusters, and pruning ratio (gamma = 0.4):
  1. Pre-HFE      : E(x)
  2. Post-DDN HFE : E(x + delta_DDN)
  3. PGD-DeltaE   : E(x + delta_PGD) - E(x) under fixed-budget L_inf-PGD
  4. DDN-DeltaE   : E(x + delta_DDN) - E(x) under 100-step minimum-L2 DDN (Ours)

Separately reports:
  - Stage 2 Cluster Identification (PASS / FAIL using each frequency cue alone)
  - Stage 3 Residual-Poison Filtering (PRR and RPR) on the fixed true benign cluster.

Example:
    python reproduce_tables/table9_frequency_cues.py \
        --attack BadNets --seed 666 \
        --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.defenses.analysis_utils as utils
from reproduce_tables.common import (
    build_attack,
    build_model,
    cluster_and_split,
    extract_epar_probes,
    get_clean_targets,
    get_dataset_config,
    load_base_datasets,
    load_checkpoint,
)


def compute_pgd_delta_hfe(model, dataset, indices, device, eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=10, batch_size=64):
    """Computes fixed-budget L_inf-PGD differential HFE for each sample index."""
    model.eval()
    loader = DataLoader(utils.MatchSampleDataset(dataset, indices), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    pgd_delta = {}

    for imgs, _, idxs in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            orig_preds = model(imgs).argmax(dim=1)
        adv = imgs.clone().detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = criterion(model(adv), orig_preds)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + alpha * grad.sign()
            adv = torch.clamp(imgs + torch.clamp(adv - imgs, -eps, eps), 0.0, 1.0).detach()

        for i, oid in enumerate(idxs.tolist()):
            pre_e = utils.fft_analysis(imgs[i])
            post_e = utils.fft_analysis(adv[i])
            pgd_delta[oid] = post_e - pre_e
    return pgd_delta


def compute_ddn_all_hfe(model, dataset, indices, device, num_classes, cache_path, batch_size=64):
    """Computes or loads Pre-HFE, Post-DDN HFE, and DDN-DeltaE for all requested indices."""
    cached = torch.load(cache_path) if osp.exists(cache_path) else {}
    missing = [i for i in indices if i not in cached or not isinstance(cached[i], dict)]
    if missing:
        loader = DataLoader(utils.MatchSampleDataset(dataset, missing), batch_size=batch_size, shuffle=False)
        results = utils.analyze_batch_with_ddn(model, loader, device, num_classes=num_classes, ddn_steps=100)
        for i, res in enumerate(results):
            cached[missing[i]] = {
                "pre_attack_hfe": res["pre_attack_hfe"],
                "post_attack_hfe": res["post_attack_hfe"],
                "freq_diff_pre_post_energy": res["freq_diff_pre_post_energy"],
            }
        os.makedirs(osp.dirname(osp.abspath(cache_path)), exist_ok=True)
        torch.save(cached, cache_path)
    return cached


def eval_stage3_prr_rpr(cand_indices, cue_map, pgd_map, bias, poisoned_set, gamma=0.4):
    h_arr = np.asarray([cue_map[i] for i in cand_indices], dtype=np.float64)
    p_arr = np.asarray([pgd_map[i] for i in cand_indices], dtype=np.float64)

    lo_h, hi_h = np.percentile(h_arr, 2), np.percentile(h_arr, 98)
    if hi_h - lo_h < 1e-6:
        lo_h, hi_h = h_arr.min(), h_arr.max()
    norm_h = np.clip((h_arr - lo_h) / (hi_h - lo_h + 1e-6), 0.0, 1.0)
    clean_h = norm_h if bias < 0 else (1.0 - norm_h)

    lo_p, hi_p = np.percentile(p_arr, 2), np.percentile(p_arr, 98)
    if hi_p - lo_p < 1e-6:
        lo_p, hi_p = p_arr.min(), p_arr.max()
    norm_p = np.clip((p_arr - lo_p) / (hi_p - lo_p + 1e-6), 1e-3, 1.0)

    order = np.argsort(-(clean_h * norm_p))
    keep_cnt = max(1, int(len(cand_indices) * (1.0 - gamma)))
    retained = [cand_indices[i] for i in order[:keep_cnt]]

    res_before = len(set(cand_indices) & poisoned_set)
    res_after = len(set(retained) & poisoned_set)
    prr = ((res_before - res_after) / res_before * 100.0) if res_before > 0 else 100.0
    rpr = (res_after / len(retained) * 100.0) if retained else 0.0
    return prr, rpr


def main():
    parser = argparse.ArgumentParser(description="Table 9: Controlled Comparison of Frequency Cues on CIFAR-10")
    parser.add_argument("--attack", type=str, required=True, choices=["BadNets", "Blended", "Refool", "WaNet", "SIG"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--gamma", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_dataset_config("CIFAR10")
    trainset, testset = load_base_datasets("CIFAR10", args.data_dir)
    orig_targets = get_clean_targets(trainset)

    attack = build_attack("CIFAR10", args.attack, trainset, testset, osp.dirname(osp.abspath(args.ckpt_path)), seed=args.seed)
    p_train, _ = attack.get_poisoned_dataset()

    model = load_checkpoint(build_model("resnet18", cfg["num_classes"]), args.ckpt_path).to(device).eval()
    probes = extract_epar_probes("CIFAR10", args.attack, p_train, orig_targets, model, args.cache_dir, args.seed, device)

    (idx0, _, _), (idx1, _, _) = cluster_and_split(probes, args.seed, device, init_anchor=True)
    poisoned_set = probes["poisoned_set"]

    # Ground-truth benign cluster (lower poison ratio) fixed across all four cues for Stage 3
    p_rate0 = len(set(idx0) & poisoned_set) / max(len(idx0), 1)
    p_rate1 = len(set(idx1) & poisoned_set) / max(len(idx1), 1)
    true_benign, true_poison = (idx0, idx1) if p_rate0 <= p_rate1 else (idx1, idx0)

    trusted = utils.get_few_shot_clean_subset(p_train, num_classes=cfg["num_classes"], shots=cfg["shots"], seed=args.seed)
    anchor_train_indices = [
        int(trusted.indices[i]) for i in range(len(trusted)) if int(trusted[i][1]) == cfg["target_class"]
    ]
    all_target_indices = list(dict.fromkeys(probes["g2_indices"] + anchor_train_indices))

    ddn_full_cache = osp.join(args.cache_dir, f"CIFAR10_{args.attack}_seed{args.seed}_ddn_full_hfe.pth")
    ddn_full = compute_ddn_all_hfe(model, p_train, all_target_indices, device, cfg["num_classes"], ddn_full_cache)
    pgd_delta = compute_pgd_delta_hfe(model, p_train, all_target_indices, device)

    cue_maps = {
        "Pre-HFE": {i: ddn_full[i]["pre_attack_hfe"] for i in all_target_indices},
        "Post-DDN HFE": {i: ddn_full[i]["post_attack_hfe"] for i in all_target_indices},
        "PGD-DeltaE": pgd_delta,
        "DDN-DeltaE (Ours)": {i: ddn_full[i]["freq_diff_pre_post_energy"] for i in all_target_indices},
    }

    print(f"\n=== Table 9 Frequency Cue Comparison [CIFAR-10 | {args.attack} | Seed {args.seed} | gamma={args.gamma}] ===")
    print(f"  {'Frequency Cue':<18} | {'Stage 2 ID':>10} | {'PRR (%)':>8} | {'RPR (%)':>8}")
    print("  " + "-" * 52)
    for cue_name, cmap in cue_maps.items():
        anc_mean = float(np.mean([cmap[i] for i in anchor_train_indices]))
        m0 = float(np.mean([cmap[i] for i in idx0]))
        m1 = float(np.mean([cmap[i] for i in idx1]))
        picked_benign = idx0 if abs(m0 - anc_mean) <= abs(m1 - anc_mean) else idx1
        id_ok = "PASS" if (picked_benign is true_benign) else "FAIL"

        bias = float(np.mean([cmap[i] for i in true_poison])) - float(np.mean([cmap[i] for i in true_benign]))
        prr, rpr = eval_stage3_prr_rpr(true_benign, cmap, probes["pgd_map"], bias, poisoned_set, gamma=args.gamma)
        print(f"  {cue_name:<18} | {id_ok:>10} | {prr:8.2f} | {rpr:8.2f}")


if __name__ == "__main__":
    main()