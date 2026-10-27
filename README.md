# EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis


![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic) ![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-DodgerBlue.svg?style=plastic) ![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic) ![CUDA 11.1](https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic) ![License GPL](https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic)




This repository provides the official PyTorch implementation of **EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis**.

## Data Preparation

Experiments are conducted on **CIFAR-10**, **GTSRB**, and **Tiny-ImageNet**:
- **CIFAR-10** and **GTSRB** are automatically downloaded to `./data` via `torchvision.datasets`.
- **Tiny-ImageNet** should be extracted to `./data/tiny-imagenet-200/`.

## Reproducing Main Experimental Results

All experiments are evaluated across three independent random seeds (`666`, `777`, `888`). The target label is `0` for CIFAR-10 and Tiny-ImageNet, and `1` for GTSRB. The poisoning rate is `10%` for poison-label attacks (BadNets, Blended, Refool, WaNet) and `5%` for clean-label SIG.

### 1. CIFAR-10

```shell
# Step 1: Train poisoned model (Attacks: BadNets, Blended, SIG, refool, Wanet)
python Attack_BadNets.py

# Step 2: Run EPAR defense (Defenses: Badnets, Blended, SIG, Refool, Wanet)
python Defense_cifar10_Badnets.py
```

### 2. GTSRB

```shell
# Step 1: Train poisoned model (--attack: BadNets | Blended | SIG | Refool | WaNet)
python Attack_GTSRB.py --attack BadNets --seed 666

# Step 2: Run EPAR defense
python Defense_GTSRB.py --attack BadNets --ckpt_path ./experiments/GTSRB_BadNets/ckpt_epoch_100.pth --seed 666
```

### 3. Tiny-ImageNet

```shell
# Step 1: Train poisoned model (--attack: BadNets | Blended | Refool | WaNet)
python Attack_TinyImageNet.py --attack BadNets --seed 666

# Step 2: Run EPAR defense
python Defense_TinyImageNet.py --attack BadNets --ckpt_path ./experiments/Tiny-ImageNet_BadNets/ckpt_epoch_100.pth --seed 666
```

## Acknowledgements

This repository is built upon [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox), an open-source toolkit for backdoor learning.
