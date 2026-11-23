# EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis

<img src="https://img.shields.io/badge/python-3.8+-DodgerBlue.svg?style=plastic" alt="Python 3.8+"> <img src="https://img.shields.io/badge/pytorch-1.13.0-DodgerBlue.svg?style=plastic" alt="PyTorch 1.13.0"> <img src="https://img.shields.io/badge/torchvision-0.14.0-DodgerBlue.svg?style=plastic" alt="torchvision 0.14.0"> <img src="https://img.shields.io/badge/cuda-11.7-DodgerBlue.svg?style=plastic" alt="CUDA 11.7"> <img src="https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic" alt="License GPL">

Official PyTorch implementation, hyperparameter configurations, statistical analysis script, and modular table-reproduction suite for **EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis**.

## Repository Structure

```text
├── core/defenses/
│   ├── epar.py                     # Core EPAR defense framework
│   ├── official_cul.py             # Clean Unlearning (CUL) exposure module
│   ├── analysis_utils.py           # 100-step min-L2 DDN, FFT-HFE, and 50-step PGD probing
│   └── AIBD.py                     # Unified-protocol AIBD (AAAI 2025) baseline module
├── Defense_cifar10_*.py            # CIFAR-10 EPAR defense entry scripts
├── Defense_GTSRB.py                # GTSRB EPAR defense entry script
├── Defense_TinyImageNet.py         # Tiny-ImageNet EPAR defense entry script
├── Defense_AIBD.py                 # Unified-protocol AIBD defense entry script
├── prepare_tiny_imagenet.py        # Tiny-ImageNet download & val_annotations.txt organizer
└── reproduce_tables/               # Modular per-table reproduction scripts
    ├── common.py                   # Shared dataset, attack, model, and probing utilities
    ├── table2_wilcoxon_stats.py    # Table 2 exact two-sided paired Wilcoxon tests (N = 14)
    ├── table3_architectures.py     # Table 3 Tiny-ImageNet multi-architecture evaluation
    ├── table6_isolation_quality.py # Table 6 CIFAR-10 TPR/FPR at gamma = 0.2 and 0.4
    ├── table7_module_necessity.py  # Table 7 w/o Refinement, w/o Partition, and Full EPAR
    ├── table8_decoupled_ablation.py# Table 8 equal-budget Feature vs. Frequency & init ablation
    ├── table9_frequency_cues.py    # Table 9 controlled comparison of frequency cues on CIFAR-10
    └── table10_gamma_sensitivity.py# Table 10 pruning ratio gamma sweep (0.2–0.5)
└── results/
    └── per_seed_results.json       # Per-seed EPAR results (Seeds 666, 777, 888) and Table 2 baselines
```

## 1. Environment & Data Preparation

```shell
pip install -r requirements.txt
```

- **CIFAR-10** and **GTSRB** download automatically to `./data` via `torchvision.datasets`.
- **Tiny-ImageNet-200**: Run `prepare_tiny_imagenet.py` to extract `tiny-imagenet-200.zip` and organize `val/val_annotations.txt` into 200 class-indexed folders under `./data/tiny-imagenet-200/val_classified/`:

```shell
python prepare_tiny_imagenet.py --data-dir ./data
```

Across all datasets, test/validation splits are reserved strictly for final ACC/ASR evaluation and never accessed during defense construction or clean-seed selection.

## 2. Dataset Hyperparameter Summary

| Dataset | Classes | Target Label | Clean Seeds / Class | CUL Optimizer & LR | Retrain Augmentation | Retrain LR Milestones | Default $\gamma$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CIFAR-10** | 10 | `0` | 30 | Adam, `1e-4` | RandomCrop(32, pad=4) + HFlip | `[75, 90]` | `0.4` |
| **GTSRB** | 43 | `1` | 30 | Adam, `1e-5` | None | `[30, 60]` | `0.4` |
| **Tiny-ImageNet** | 200 | `0` | 10 | Adam, `1e-5` | None | `[75, 90]` | `0.4` |

## 3. Reproducing Main Tables

### Table 2: Main Defense Performance & Wilcoxon Significance Tests
```shell
# EPAR on CIFAR-10 / GTSRB / Tiny-ImageNet (Seeds: 666, 777, 888)
python Attack_BadNets.py && python Defense_cifar10_Badnets.py
python Attack_GTSRB.py --attack BadNets --seed 666 && python Defense_GTSRB.py --attack BadNets --ckpt_path ./experiments/GTSRB_BadNets/ckpt_epoch_100.pth --seed 666
python Attack_TinyImageNet.py --attack BadNets --seed 666 && python Defense_TinyImageNet.py --attack BadNets --ckpt_path ./experiments/Tiny-ImageNet_BadNets/ckpt_epoch_100.pth --seed 666

# Unified-protocol AIBD baseline
python Defense_AIBD.py --dataset CIFAR10 --attack BadNets --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth --seed 666

# Exact two-sided paired Wilcoxon signed-rank tests (N = 14)
python reproduce_tables/table2_wilcoxon_stats.py --results_json results/per_seed_results.json
```

### Table 3: Multi-Architecture Evaluation on Tiny-ImageNet
```shell
python reproduce_tables/table3_architectures.py --arch vgg19_bn --attack BadNets --ckpt_path ./experiments/Tiny-ImageNet_BadNets_vgg19_bn/ckpt_epoch_100.pth --seed 666
```

### Table 6: Sample Isolation Quality on CIFAR-10 ($\gamma = 0.2$ and $\gamma = 0.4$)
```shell
python reproduce_tables/table6_isolation_quality.py --attack BadNets --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth --seed 666
```

### Table 7: Module Necessity Ablation (`w/o Refinement`, `w/o Partition`, `Full`)
```shell
python reproduce_tables/table7_module_necessity.py --attack BadNets --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth --seed 666 --retrain
```

### Table 8: Equal-Budget ($\gamma = 0.4$) Decoupled Feature vs. Frequency Ablation
```shell
python reproduce_tables/table8_decoupled_ablation.py --dataset TinyImageNet --attack BadNets --ckpt_path ./experiments/Tiny-ImageNet_BadNets/ckpt_epoch_100.pth --seed 666
```
### Table 9: Controlled Comparison of Frequency Cues on CIFAR-10
```shell
python reproduce_tables/table9_frequency_cues.py --attack BadNets --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth --seed 666
```

### Table 10: Pruning Ratio $\gamma$ Sensitivity
```shell
python reproduce_tables/table10_gamma_sensitivity.py --attack BadNets --ckpt_path ./experiments/CIFAR-10_BadNets/ckpt_epoch_100.pth --seed 666 --retrain
```

## Acknowledgements

This repository is built upon [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox), an open-source toolkit for backdoor learning.
