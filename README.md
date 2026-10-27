# EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis

![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic) ![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-DodgerBlue.svg?style=plastic) ![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic) ![CUDA 11.1](https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic) ![License GPL](https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic)

This repository provides the PyTorch implementation of our work: **EPAR: Anchor-Guided Adaptive Backdoor Defense via Adversarial Frequency Analysis**. 


## Data Preparation

The experiments are mainly conducted on CIFAR-10 and Tiny-ImageNet datasets. Please ensure the datasets are downloaded and extracted to your specified data directories before running the attack scripts.

## Backdoor Attack

Run the following commands to generate the poisoned datasets and train the infected models.

- **BadNets on CIFAR-10:**
  ```shell
  python Attack_BadNets.py
  ```

- **Blended Attack on Tiny-ImageNet:**
  ```shell
  python Attack_Blended.py
  ```

## Backdoor Defense

Once the compromised models and datasets are prepared, run the following commands to evaluate our EPAR defense.

- **Defend against BadNets on CIFAR-10:**
  ```shell
  python Defense_cifar10_Badnets.py
  ```

- **Defend against Blended Attack on Tiny-ImageNet:**
  ```shell
  python Defense_Tiny-imagenet_Blended.py
  ```

## Acknowledgements

This repository is mainly based on [BackdoorBox](https://github.com/SCLBD/BackdoorBox), a comprehensive and standardized open-source toolkit for backdoor learning. Thanks for their wonderful work!
