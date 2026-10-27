# EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis







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
