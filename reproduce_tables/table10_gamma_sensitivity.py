#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 10 Reproduction: Impact of the Pruning Ratio gamma on CIFAR-10.

Sweeps gamma in {0.2, 0.3, 0.4, 0.5} on CIFAR-10 (BadNets, Blended, Refool) and reports
isolation TPR/FPR and optional 100-epoch retrained ACC / ASR.

Example:
    python reproduce_tables/table10_gamma_sensitivity.py \
        --attack BadNets --seed 666 \
        --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth --retrain
"""

import argparse
import os.path as osp
import sys
import torch

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.utils.accuracy import filter_poisoned_testset
from reproduce_tables.common import (
    build_attack,
    build_model,
    cluster_and_split,
    extract_epar_probes,
    get_clean_targets,
    get_dataset_config,
    load_base_datasets,
    load_checkpoint,
    retrain_and_eval,
    stage3_prune,
)


def main():
    parser = argparse.ArgumentParser(description="Table 10: Pruning Ratio gamma Sensitivity on CIFAR-10")
    parser.add_argument("--attack", type=str, required=True, choices=["BadNets", "Blended", "Refool"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--retrain", action="store_true", help="Run 100-epoch retraining for each gamma")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_dataset_config("CIFAR10")
    trainset, testset = load_base_datasets("CIFAR10", args.data_dir)
    orig_targets = get_clean_targets(trainset)

    attack = build_attack("CIFAR10", args.attack, trainset, testset, osp.dirname(osp.abspath(args.ckpt_path)), seed=args.seed)
    p_train, p_test = attack.get_poisoned_dataset()
    p_test = filter_poisoned_testset(p_test, testset, target_class=cfg["target_class"])

    model = load_checkpoint(build_model("resnet18", cfg["num_classes"]), args.ckpt_path)
    probes = extract_epar_probes("CIFAR10", args.attack, p_train, orig_targets, model, args.cache_dir, args.seed, device)

    (idx0, hfe0, d0), (idx1, hfe1, d1) = cluster_and_split(probes, args.seed, device, init_anchor=True)
    s0 = abs(hfe0 - probes["hfe_anchor"]) * max(d0, 1e-4)
    s1 = abs(hfe1 - probes["hfe_anchor"]) * max(d1, 1e-4)
    cand, mc, mp = (idx0, hfe0, hfe1) if s0 <= s1 else (idx1, hfe1, hfe0)

    print(f"\n=== Table 10 Pruning Ratio Sensitivity [CIFAR-10 | {args.attack} | Seed {args.seed}] ===")
    for gamma in args.gammas:
        tp, fp, tpr, fpr, kept = stage3_prune(cand, probes, mp - mc, s3_mode="combined", gamma=gamma)
        line = f"  gamma={gamma:.1f} | TPR={tpr:6.2f}% FPR={fpr:5.2f}%"
        if args.retrain:
            acc, asr = retrain_and_eval("CIFAR10", p_train, p_test, testset, kept, args.seed, device)
            line += f" | ACC={acc:6.2f}% ASR={asr:6.2f}%"
        print(line)


if __name__ == "__main__":
    main()
