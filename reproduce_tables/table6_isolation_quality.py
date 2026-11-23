#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 6 Reproduction: Sample Isolation Quality (TPR / FPR) on CIFAR-10.

Evaluates EPAR at gamma = 0.2 and gamma = 0.4 under the unified CIFAR-10 protocol
(for AIBD's same-protocol TPR/FPR, run `python Defense_AIBD.py --dataset CIFAR10 ...`).

Example:
    python reproduce_tables/table6_isolation_quality.py \
        --attack BadNets --seed 666 \
        --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth
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
    parser = argparse.ArgumentParser(description="Table 6: CIFAR-10 Sample Isolation Quality (TPR/FPR)")
    parser.add_argument("--attack", type=str, required=True, choices=["BadNets", "Blended", "SIG"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_dataset_config("CIFAR10")
    trainset, testset = load_base_datasets("CIFAR10", args.data_dir)
    orig_targets = get_clean_targets(trainset)

    attack = build_attack("CIFAR10", args.attack, trainset, testset, osp.dirname(osp.abspath(args.ckpt_path)), seed=args.seed)
    p_train, _ = attack.get_poisoned_dataset()

    model = load_checkpoint(build_model("resnet18", cfg["num_classes"]), args.ckpt_path)
    probes = extract_epar_probes("CIFAR10", args.attack, p_train, orig_targets, model, args.cache_dir, args.seed, device)

    (idx0, hfe0, d0), (idx1, hfe1, d1) = cluster_and_split(probes, args.seed, device, init_anchor=True)
    s0 = abs(hfe0 - probes["hfe_anchor"]) * max(d0, 1e-4)
    s1 = abs(hfe1 - probes["hfe_anchor"]) * max(d1, 1e-4)
    cand, mc, mp = (idx0, hfe0, hfe1) if s0 <= s1 else (idx1, hfe1, hfe0)

    print(f"\n=== Table 6 Isolation Quality [CIFAR-10 | {args.attack} | Seed {args.seed}] ===")
    for gamma in (0.2, 0.4):
        tp, fp, tpr, fpr, _ = stage3_prune(cand, probes, mp - mc, s3_mode="combined", gamma=gamma)
        print(f"  EPAR (gamma={gamma:.1f}) | TP={tp:<5} FP={fp:<4} | TPR={tpr:6.2f}% | FPR={fpr:5.2f}%")


if __name__ == "__main__":
    main()
