#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download, extract, and organize Tiny-ImageNet-200 validation split.

Usage:
    python prepare_tiny_imagenet.py --data-dir ./data
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = [
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
]


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        done = block_num * block_size
        if total_size > 0:
            percent = min(100.0, done * 100.0 / total_size)
            print(
                f"\rDownloading: {percent:6.2f}% "
                f"({done / 1024 / 1024:7.1f}/{total_size / 1024 / 1024:7.1f} MB)",
                end="",
                flush=True,
            )
        else:
            print(f"\rDownloaded: {done / 1024 / 1024:7.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, output, reporthook=progress)
    print()


def ensure_zip(zip_path: Path) -> None:
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"[Skip] Archive already exists: {zip_path}")
        return

    last_error: Exception | None = None
    for url in URLS:
        try:
            print(f"[Download] {url}")
            download(url, zip_path)
            print(f"[Done] Saved to: {zip_path}")
            return
        except Exception as exc:
            last_error = exc
            print(f"\n[Failed] {url}\nReason: {exc}")
            if zip_path.exists():
                zip_path.unlink()

    raise RuntimeError(
        "Download failed. Please check network connectivity or manually place "
        "tiny-imagenet-200.zip into --data-dir."
    ) from last_error


def extract(zip_path: Path, data_dir: Path, dataset_root: Path) -> None:
    if dataset_root.exists():
        print(f"[Skip] Dataset directory already exists: {dataset_root}")
        return

    print(f"[Extract] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)
    print(f"[Done] Extracted to: {dataset_root}")


def prepare_val(dataset_root: Path) -> Path:
    val_dir = dataset_root / "val"
    image_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"
    output_dir = dataset_root / "val_classified"

    if not annotations.exists():
        raise FileNotFoundError(f"Not found: {annotations}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, str]] = []
    with annotations.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                records.append((parts[0], parts[1]))

    created = 0
    skipped = 0
    print(f"[Organize] Validation split ({len(records)} images) -> {output_dir}")

    for index, (image_name, class_name) in enumerate(records, start=1):
        source = image_dir / image_name
        class_dir = output_dir / class_name
        destination = class_dir / image_name
        class_dir.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            skipped += 1
        else:
            if not source.exists():
                raise FileNotFoundError(f"Validation image missing: {source}")
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            created += 1

        if index % 1000 == 0 or index == len(records):
            print(f"\rProgress: {index}/{len(records)}", end="", flush=True)

    print()
    print(f"[Done] Created {created}, skipped {skipped}")
    return output_dir


def verify(dataset_root: Path, val_root: Path) -> None:
    train_root = dataset_root / "train"
    train_classes = [p for p in train_root.iterdir() if p.is_dir()]
    val_classes = [p for p in val_root.iterdir() if p.is_dir()]
    train_count = sum(1 for _ in train_root.rglob("*.JPEG"))
    val_count = sum(1 for _ in val_root.rglob("*.JPEG"))

    print("\n========== Tiny-ImageNet Verification ==========")
    print(f"Train classes: {len(train_classes)} (expected 200)")
    print(f"Val classes  : {len(val_classes)} (expected 200)")
    print(f"Train images : {train_count} (expected 100000)")
    print(f"Val images   : {val_count} (expected 10000)")
    print(f"Train dir    : {train_root}")
    print(f"Val dir      : {val_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tiny-ImageNet-200 for EPAR")
    parser.add_argument("--data-dir", default="./data", help="Root data directory (default: ./data)")
    parser.add_argument("--keep-zip", action="store_true", help="Keep tiny-imagenet-200.zip after extraction")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    zip_path = data_dir / "tiny-imagenet-200.zip"
    dataset_root = data_dir / "tiny-imagenet-200"

    try:
        print(f"Data directory: {data_dir}")
        if not dataset_root.exists():
            ensure_zip(zip_path)
            extract(zip_path, data_dir, dataset_root)
        val_root = prepare_val(dataset_root)
        verify(dataset_root, val_root)

        if not args.keep_zip and zip_path.exists():
            zip_path.unlink()
            print(f"\n[Cleanup] Removed archive: {zip_path}")

        print("\nTiny-ImageNet-200 is ready.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[Error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
