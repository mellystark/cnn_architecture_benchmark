from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def _extract_labels(dataset):
    labels = getattr(dataset, "_labels", None)
    if labels is None:
        raise RuntimeError(
            "OxfordIIITPet label listesi bulunamadı. Torchvision API'si değişmiş olabilir."
        )
    return np.asarray(labels)


def build_dataloaders(
    data_root,
    image_size=224,
    batch_size=32,
    num_workers=4,
    val_ratio=0.20,
    seed=42,
):
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    train_transform, eval_transform = build_transforms(image_size)

    trainval_train_tf = datasets.OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=True,
    )

    trainval_eval_tf = datasets.OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        transform=eval_transform,
        download=True,
    )

    test_dataset = datasets.OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        transform=eval_transform,
        download=True,
    )

    labels = _extract_labels(trainval_train_tf)
    all_indices = np.arange(len(labels))

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    train_dataset = Subset(trainval_train_tf, train_indices)
    val_dataset = Subset(trainval_eval_tf, val_indices)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    classes = list(getattr(trainval_train_tf, "classes", []))
    if not classes:
        classes = [f"class_{i}" for i in range(int(labels.max()) + 1)]

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "classes": classes,
        "sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    }
