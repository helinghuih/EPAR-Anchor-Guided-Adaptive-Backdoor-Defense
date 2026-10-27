# EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis

[https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic] [https://img.shields.io/badge/pytorch-1.8.0-DodgerBlue.svg?style=plastic] [https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic] [https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic] [https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic]

This repository provides the official PyTorch implementation of **EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis**.

## Overview

EPAR is a four-stage backdoor defense framework consisting of:
1. **Exposure and Partition (`core/defenses/official_cul.py`)**: Performs gradient-ascent candidate unlearning on the trusted clean reference set $\mathcal{D}_{seed}$ and infers the target class $t$ via excess label retention.
2. **Anchor-Guided Identification (`core/defenses/epar.py`)**: Reuses the target-class clean seeds as anchors $\mathcal{D}_{anchor}^{t}$ and identifies the poisoned cluster in $G_2$ via joint feature-frequency distance.
3. **Adaptive Refinement (`core/defenses/analysis_utils.py`)**: Computes 100-step minimum-$\ell_2$-norm DDN differential high-frequency energy ($\text{HFE}_{\text{diff}}$) and 50-step $\ell_\infty$-PGD flip steps to filter residual poisoned samples at pruning ratio $\rho = 0.4$.
4. **Model Retraining (`core/defenses/epar.py`)**: Retrains a purified model from scratch on the sanitized training set.

## Data Preparation

Experiments are conducted on **CIFAR-10** ($32 \times 32$, 10 classes), **GTSRB** ($32 \times 32$, 43 classes), and **Tiny-ImageNet** ($64 \times 64$, 200 classes).
- CIFAR-10 and GTSRB are automatically downloaded to `./data` via `torchvision.datasets`.
- For Tiny-ImageNet, extract the dataset to `./data/tiny-imagenet-200/`.

## Reproducing Main Experimental Results (Table 2)

All experiments are evaluated across three independent random seeds (`666`, `777`, `888`). Target label $y_t = 0$ for CIFAR-10 and Tiny-ImageNet, and $y_t = 1$ for GTSRB. The poisoning rate is $\gamma = 10\%$ for poison-label attacks (BadNets, Blended, Refool, WaNet) and $\gamma = 5\%$ for clean-label SIG.

### 1. CIFAR-10 (30 Clean Seeds/Class, Total 300)

**Step 1: Train Poisoned Models**
```shell
python Attack_BadNets.py
python Attack_Blended.py
python Attack_SIG.py
python Attack_refool.py
python Attack_Wanet.py
