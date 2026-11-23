#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 8 Reproduction: Equal-Budget (gamma = 0.4) Decoupled Ablation Across Three Datasets.

Isolates:
  1. Feature   (Stage 2 cosine distance d_cos + Stage 3 PGD robustness R)
  2. Frequency (Stage 2 differential HFE DeltaE + Stage 3 differential HFE DeltaE)
  3. EPAR      (Stage 2 joint distance + Stage 3 DeltaE * R)
  4. Optional anchor-guided vs. random K-Means centroid initialization comparison.

Example:
    python reproduce_tables/table8_decoupled_ablation.py \
        --dataset TinyImageNet --attack BadNets --seed 666 \
        --ckpt_path ./experiments/Tiny-ImageNet_BadNets/ckpt_epoch_100.pth
"""

import argparse
import os.path as osp
import sys
import torch

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproduce_tables.common import (
    build_attack,
    build_model,
    cluster_and_split,
    extract_epar_probes,
    get_clean_targets,
    get_dataset_config,
    load_base_datasets,
    load_checkpoint,
    stage3_prune,
)


def main():
    parser = argparse.ArgumentParser(description="Table 8: Equal-Budget Decoupled Feature vs. Frequency Ablation")
    parser.add_argument("--dataset", type=str, required=True, choices=["CIFAR10", "GTSRB", "TinyImageNet"])
    parser.add_argument("--attack", type=str, required=True, choices=["BadNets", "Blended", "WaNet"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--gamma", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_dataset_config(args.dataset)
    trainset, testset = load_base_datasets(args.dataset, args.data_dir)
    orig_targets = get_clean_targets(trainset)

    attack = build_attack(args.dataset, args.attack, trainset, testset, osp.dirname(osp.abspath(args.ckpt_path)), seed=args.seed)
    p_train, _ = attack.get_poisoned_dataset()

    model = load_checkpoint(build_model("resnet18", cfg["num_classes"]), args.ckpt_path)
    probes = extract_epar_probes(args.dataset, args.attack, p_train, orig_targets, model, args.cache_dir, args.seed, device)

    (idx0, hfe0, d0), (idx1, hfe1, d1) = cluster_and_split(probes, args.seed, device, init_anchor=True)
    dh0, dh1 = abs(hfe0 - probes["hfe_anchor"]), abs(hfe1 - probes["hfe_anchor"])

    print(f"\n=== Table 8 Decoupled Ablation [{args.dataset} | {args.attack} | Seed {args.seed} | gamma={args.gamma}] ===")
    variants = [
        ("Feature   (S2: d_cos, S3: PGD R)",    d0,                     d1,                     "pgd_only"),
        ("Frequency (S2: DeltaE, S3: DeltaE)",  dh0,                    dh1,                    "freq_only"),
        ("EPAR      (S2: Joint, S3: DeltaE*R)", dh0 * max(d0, 1e-4),    dh1 * max(d1, 1e-4),    "combined"),
    ]
    for label, s0, s1, s3_mode in variants:
        cand, mc, mp = (idx0, hfe0, hfe1) if s0 <= s1 else (idx1, hfe1, hfe0)
        tp, fp, tpr, fpr, _ = stage3_prune(cand, probes, mp - mc, s3_mode=s3_mode, gamma=args.gamma)
        print(f"  {label:<38} | TP={tp:<5} FP={fp:<4} | TPR={tpr:6.2f}% | FPR={fpr:5.2f}%")

    # Anchor vs. Random K-Means initialization comparison
    print("\n--- Anchor-Guided vs. Random K-Means Initialization ---")
    for init_flag, name in [(True, "Anchor-Init"), (False, "Random-Init")]:
        (i0, h0, c0), (i1, h1, c1) = cluster_and_split(probes, args.seed, device, init_anchor=init_flag)
        js0 = abs(h0 - probes["hfe_anchor"]) * max(c0, 1e-4)
        js1 = abs(h1 - probes["hfe_anchor"]) * max(c1, 1e-4)
        cand, mc, mp = (i0, h0, h1) if js0 <= js1 else (i1, h1, h0)
        tp, fp, tpr, fpr, _ = stage3_prune(cand, probes, mp - mc, s3_mode="combined", gamma=args.gamma)
        print(f"  {name:<12} | TP={tp:<5} FP={fp:<4} | TPR={tpr:6.2f}% | FPR={fpr:5.2f}%")


if __name__ == "__main__":
    main()
