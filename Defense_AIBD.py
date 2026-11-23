#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry script for AIBD (AAAI 2025) baseline evaluation under EPAR's unified protocol.

Example:
    python Defense_AIBD.py \
        --dataset CIFAR10 --attack BadNets --seed 666 \
        --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth
"""

import argparse
import os.path as osp
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.defenses.AIBD import AIBD
from reproduce_tables.common import (
    build_attack,
    build_model,
    get_clean_targets,
    get_dataset_config,
    load_base_datasets,
    load_checkpoint,
)
from core.utils.accuracy import filter_poisoned_testset


def main():
    parser = argparse.ArgumentParser(description="Run AIBD defense under EPAR's unified protocol")
    parser.add_argument("--dataset", type=str, required=True, choices=["CIFAR10", "GTSRB", "TinyImageNet"])
    parser.add_argument("--attack", type=str, required=True, choices=["BadNets", "Blended", "Refool", "WaNet", "SIG"])
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pretrained poisoned checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--q_strategy", type=str, default="progressive", choices=["progressive", "fixed"])
    parser.add_argument("--scope", type=str, default="target", choices=["target", "global"])
    parser.add_argument("--target_mode", type=str, default="oracle", choices=["oracle", "vote"])
    parser.add_argument("--use_retrain_aug", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--drop_acc_threshold", type=str, default="auto")
    parser.add_argument("--retrain_epochs", type=int, default=15)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = get_dataset_config(args.dataset)
    trainset, testset = load_base_datasets(args.dataset, args.data_dir)
    orig_targets = get_clean_targets(trainset)

    ckpt_dir = osp.dirname(osp.abspath(args.ckpt_path))
    attack = build_attack(args.dataset, args.attack, trainset, testset, ckpt_dir, seed=args.seed)
    p_train, p_test = attack.get_poisoned_dataset()
    p_test = filter_poisoned_testset(p_test, testset, target_class=cfg["target_class"])
    poisoned_set = set(int(i) for i in p_train.poisoned_set)

    given_labels = [
        cfg["target_class"] if (i in poisoned_set or int(orig_targets[i]) == cfg["target_class"]) else int(orig_targets[i])
        for i in range(len(p_train))
    ]

    model = build_model("resnet18", cfg["num_classes"])
    load_checkpoint(model, args.ckpt_path)

    use_aug = cfg["default_retrain_aug"] if args.use_retrain_aug == "auto" else (args.use_retrain_aug == "true")
    drop_thr = cfg["default_drop_acc_thr"] if args.drop_acc_threshold == "auto" else float(args.drop_acc_threshold)
    q_list = (
        [0.05] if (args.q_strategy == "fixed" and args.attack == "SIG") else
        [0.10] if args.q_strategy == "fixed" else
        [0.05, 0.04, 0.03, 0.02, 0.01] if args.attack == "SIG" else
        [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01]
    )

    defense = AIBD(
        model=model,
        poisoned_trainset=p_train,
        poisoned_testset=p_test,
        clean_testset=testset,
        num_classes=cfg["num_classes"],
        target_class=cfg["target_class"],
        seed=args.seed,
    )
    res = defense.train({
        "given_labels": given_labels,
        "poisoned_set": poisoned_set,
        "cache_path": osp.join(args.cache_dir, f"{args.dataset}_{args.attack}_seed{args.seed}_aibd_pgd.pth"),
        "q_list": q_list,
        "q_strategy": args.q_strategy,
        "scope": args.scope,
        "target_mode": args.target_mode,
        "retrain_transform": cfg["retrain_schedule"]["transform"] if use_aug else None,
        "retrain_epochs": args.retrain_epochs,
        "drop_acc_threshold": drop_thr,
    })

    print(
        f"\n[AIBD | {args.dataset} {args.attack} Seed {args.seed}] "
        f"q={res['q']*100:.1f}% | TP={res['tp']} FP={res['fp']} | "
        f"TPR={res['tpr']:.2f}% FPR={res['fpr']:.2f}% | "
        f"ACC={res['acc']:.2f}% ASR={res['asr']:.2f}%"
    )


if __name__ == "__main__":
    main()
