import torch
import numpy as np
from torch.utils.data import Subset


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def filter_poisoned_testset(poisoned_dataset, clean_dataset, target_class):
    if hasattr(clean_dataset, 'targets'):
        clean_targets = np.array(clean_dataset.targets)
    elif hasattr(clean_dataset, 'labels'):
        clean_targets = np.array(clean_dataset.labels)
    else:
        clean_targets = np.array([y for _, y in clean_dataset])
    non_target_indices = np.where(clean_targets != target_class)[0]
    excluded_count = len(poisoned_dataset) - len(non_target_indices)
    print(f"\n[ASR Filter] Target Class: {target_class}")
    print(f"Original Size: {len(poisoned_dataset)} -> Filtered Size: {len(non_target_indices)}")
    print(f"Excluded {excluded_count} clean target samples for strict ASR calculation.\n")

    subset = Subset(poisoned_dataset, non_target_indices)

    if hasattr(poisoned_dataset, 'poisoned_transform'):
        subset.poisoned_transform = poisoned_dataset.poisoned_transform

    if hasattr(poisoned_dataset, 'poisoned_target_transform'):
        subset.poisoned_target_transform = poisoned_dataset.poisoned_target_transform

    return subset