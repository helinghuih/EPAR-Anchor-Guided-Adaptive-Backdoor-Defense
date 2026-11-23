#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 7 Reproduction: Ablation Study on Module Necessity on CIFAR-10.

Compares:
  1. w/o Refinement (Stage 2 clustering only)
  2. w/o Partition  (Stage 3 pruning at gamma = 0.4 applied directly to D_t)
  3. EPAR (Full)

Explicitly logs |D_p|, |D_t|, nominal discard capacity K = floor(|D_t| * gamma),
actual removed count N_removed = TP + FP, TPR = TP / |D_p|, FPR, and optional
100-epoch retrained ACC / ASR.

Example:
    python reproduce_tables/table7_module_necessity.py \
        --attack BadNets --seed 666 \
        --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth --retrain
"""

import argparse
import os.path as osp
import sys

import numpy as np
import torch

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.utils.accuracy import filter_poisoned_testset
from reproduce_tables.common import (
    build_attack,
    build_model,
    cluster_and_split,
    compute_tpr_fpr,
    extract_epar_probes,
    get_clean_targets,
    get_dataset_config,
    load_base_datasets,
    load_checkpoint,
    retrain_and_eval,
    stage3_prune,
)


def main():
    parser = argparse.ArgumentParser(description="Table 7: Module Necessity Ablation on CIFAR-10")
    parser.add_argument("--attack", type=str, required=True, choices=["BadNets", "Blended", "WaNet"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--retrain", action="store_true", help="Run 100-epoch retraining to report ACC/ASR")
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

    # 1. w/o Refinement
    kept_wo_ref = list(dict.fromkeys(probes["non_g2_indices"] + cand + probes["trusted_indices"]))
    tp_r, fp_r, tpr_r, fpr_r = compute_tpr_fpr(kept_wo_ref, probes["total_n"], probes["poisoned_set"])

    # 2. w/o Partition (Stage 3 applied directly to entire D_t)
    g2 = probes["g2_indices"]
    k_nominal = int(np.floor(len(g2) * args.gamma))
    bias_dt = float(np.mean([probes["get_hfe"](i) for i in g2])) - probes["hfe_anchor"]
    tp_p, fp_p, tpr_p, fpr_p, kept_wo_part = stage3_prune(g2, probes, bias_dt, s3_mode="combined", gamma=args.gamma)

    # 3. EPAR (Full)
    tp_f, fp_f, tpr_f, fpr_f, kept_full = stage3_prune(cand, probes, mp - mc, s3_mode="combined", gamma=args.gamma)

    print(f"\n=== Table 7 Module Necessity [CIFAR-10 | {args.attack} | Seed {args.seed}] ===")
    print(f"  |D_p|={len(probes['poisoned_set'])}, |D_t|={len(g2)}, K_nominal=floor(|D_t|*{args.gamma})={k_nominal} (TPR ceiling={k_nominal/len(probes['poisoned_set'])*100:.2f}%)")
    print(f"  w/o Refinement | N_removed={tp_r+fp_r:<5} (TP={tp_r}, FP={fp_r}) | TPR={tpr_r:6.2f}% | FPR={fpr_r:5.2f}%")
    print(f"  w/o Partition  | N_removed={tp_p+fp_p:<5} (TP={tp_p}, FP={fp_p}) | TPR={tpr_p:6.2f}% | FPR={fpr_p:5.2f}%")
    print(f"  EPAR (Full)    | N_removed={tp_f+fp_f:<5} (TP={tp_f}, FP={fp_f}) | TPR={tpr_f:6.2f}% | FPR={fpr_f:5.2f}%")

    if args.retrain:
        for name, kept in [("w/o Refinement", kept_wo_ref), ("w/o Partition", kept_wo_part), ("EPAR (Full)", kept_full)]:
            acc, asr = retrain_and_eval("CIFAR10", p_train, p_test, testset, kept, args.seed, device)
            print(f"  [Retrained {name:<14}] ACC={acc:6.2f}% | ASR={asr:6.2f}%")


if __name__ == "__main__":
    main()
