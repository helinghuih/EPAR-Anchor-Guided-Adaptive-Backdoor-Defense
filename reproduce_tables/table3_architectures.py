#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 3 Reproduction: Robustness to Different Model Architectures on Tiny-ImageNet.

Supports ResNet18, VGG19-BN, MobileNetV2, and DenseNet121 under EPAR's unified protocol.

Example:
    python reproduce_tables/table3_architectures.py \
        --arch vgg19_bn --attack BadNets --seed 666 \
        --ckpt_path ./experiments/Tiny-ImageNet_BadNets_vgg19_bn/ckpt_epoch_100.pth
"""

import argparse
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.defenses.analysis_utils as utils
from core.defenses.epar import EPAR
from core.utils.accuracy import filter_poisoned_testset
from reproduce_tables.common import (
    build_attack,
    build_model,
    get_dataset_config,
    load_base_datasets,
    load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Table 3: Tiny-ImageNet Multi-Architecture EPAR Evaluation")
    parser.add_argument("--arch", type=str, required=True, choices=["resnet18", "vgg19_bn", "mobilenetv2", "densenet121"])
    parser.add_argument("--attack", type=str, required=True, choices=["BadNets", "Blended", "Refool", "WaNet"])
    parser.add_argument("--ckpt_path", type=str, required=True, help="Pretrained poisoned checkpoint for --arch")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = get_dataset_config("TinyImageNet")
    trainset, testset = load_base_datasets("TinyImageNet", args.data_dir)
    ckpt_dir = osp.dirname(osp.abspath(args.ckpt_path))

    attack = build_attack("TinyImageNet", args.attack, trainset, testset, ckpt_dir, seed=args.seed, arch=args.arch)
    p_train, p_test = attack.get_poisoned_dataset()
    p_test = filter_poisoned_testset(p_test, testset, target_class=cfg["target_class"])

    trusted_clean_set = utils.get_few_shot_clean_subset(
        p_train, num_classes=cfg["num_classes"], shots=cfg["shots"], seed=args.seed
    )

    model_poisoned = build_model(args.arch, cfg["num_classes"])
    load_checkpoint(model_poisoned, args.ckpt_path)

    schedule = {
        "device": "GPU",
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "save_dir": "experiments/EPAR-defense-arch",
        "experiment_name": f"{args.arch}_TinyImageNet_{args.attack}",
        "dataset_name": "TinyImageNet",
        "attack_name": args.attack,
        "cache_tag": f"TinyImageNet_{args.arch}_{args.attack}_seed{args.seed}",
        "drop_ratio": 0.4,
        "cul_schedule": {
            "batch_size": 128,
            "num_workers": 4,
            "epochs": 20,
            "lr": cfg["cul_lr"],
            "weight_decay": 5e-4,
            "schedule": [20, 20],
            "max_unlearn_loss": 7.0,
            "shots": cfg["shots"],
        },
        "retrain_schedule": cfg["retrain_schedule"],
    }

    defense = EPAR(
        model=model_poisoned,
        loss=nn.CrossEntropyLoss(),
        poisoned_trainset=p_train,
        poisoned_testset=p_test,
        clean_testset=testset,
        poisoned_retrainset=p_train,
        trusted_clean_set=trusted_clean_set,
        seed=args.seed,
        num_classes=cfg["num_classes"],
        target_class=cfg["target_class"],
    )
    defense.train(schedule, model_factory=lambda: build_model(args.arch, cfg["num_classes"]))


if __name__ == "__main__":
    main()
