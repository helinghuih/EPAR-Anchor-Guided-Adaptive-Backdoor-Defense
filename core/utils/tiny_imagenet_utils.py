
from __future__ import annotations

from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor


def build_tiny_imagenet_views(
    root: str | Path = "./data/tiny-imagenet-200",
):

    root = Path(root).expanduser().resolve()
    train_root = root / "train"
    val_root = root / "val_classified"

    if not train_root.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet training directory not found: {train_root}"
        )

    if not val_root.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet validation directory not found: {val_root}. "
            "Run prepare_tiny_imagenet.py first."
        )

    # Unified protocol: no random crop or random horizontal flip.
    train_transform = Compose([
        ToTensor(),
    ])

    analysis_transform = Compose([
        ToTensor(),
    ])

    retrain_transform = Compose([
        ToTensor(),
    ])

    test_transform = Compose([
        ToTensor(),
    ])

    train_dataset = ImageFolder(
        root=str(train_root),
        transform=train_transform,
    )

    analysis_dataset = ImageFolder(
        root=str(train_root),
        transform=analysis_transform,
    )

    retrain_dataset = ImageFolder(
        root=str(train_root),
        transform=retrain_transform,
    )

    validation_dataset = ImageFolder(
        root=str(val_root),
        transform=test_transform,
    )
    datasets = (
        train_dataset,
        analysis_dataset,
        retrain_dataset,
        validation_dataset,
    )

    reference_class_to_idx = train_dataset.class_to_idx

    for dataset in datasets[1:]:
        if dataset.class_to_idx != reference_class_to_idx:
            raise RuntimeError(
                "Tiny-ImageNet class mappings differ between dataset views."
            )

    reference_samples = train_dataset.samples

    for dataset in (analysis_dataset, retrain_dataset):
        if dataset.samples != reference_samples:
            raise RuntimeError(
                "Tiny-ImageNet training sample order differs between views."
            )

    if len(train_dataset.classes) != 200:
        raise RuntimeError(
            f"Expected 200 classes, got {len(train_dataset.classes)}."
        )

    if len(train_dataset) != 100000:
        raise RuntimeError(
            f"Expected 100000 training images, got {len(train_dataset)}."
        )

    if len(validation_dataset) != 10000:
        raise RuntimeError(
            f"Expected 10000 validation images, got "
            f"{len(validation_dataset)}."
        )

    return {
        "train": train_dataset,
        "analysis": analysis_dataset,
        "retrain": retrain_dataset,
        "validation": validation_dataset,
    }
