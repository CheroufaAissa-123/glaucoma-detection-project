from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder


@dataclass(frozen=True)
class DataPreparationConfig:
    data_root: Path
    image_size: int = 224
    batch_size: int = 32
    train_split: float = 0.8
    num_workers: int = 0
    clahe_train_probability: float = 0.65
    clahe_eval_probability: float = 0.5
    random_seed: int = 42


class AlbumentationsTransform:
    """Adaptateur pour utiliser Albumentations avec les datasets torchvision."""

    def __init__(self, pipeline: A.Compose) -> None:
        self.pipeline = pipeline

    def __call__(self, image):
        transformed = self.pipeline(image=np.array(image))
        return transformed["image"]


def build_train_transform(image_size: int, clahe_probability: float) -> AlbumentationsTransform:
    pipeline = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=clahe_probability),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.Rotate(limit=40, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=20, p=0.4),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.25),
            A.CoarseDropout(max_holes=10, max_height=32, max_width=32, fill_value=0, p=0.35),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return AlbumentationsTransform(pipeline)


def build_eval_transform(image_size: int, clahe_probability: float) -> AlbumentationsTransform:
    pipeline = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=clahe_probability),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return AlbumentationsTransform(pipeline)


def create_dataloaders(config: DataPreparationConfig) -> dict:
    train_transform = build_train_transform(config.image_size, config.clahe_train_probability)
    eval_transform = build_eval_transform(config.image_size, config.clahe_eval_probability)

    train_source = ImageFolder(Path(config.data_root) / "train", transform=train_transform)
    eval_source = ImageFolder(Path(config.data_root) / "train", transform=eval_transform)
    generator = torch.Generator().manual_seed(config.random_seed)

    num_samples = len(train_source)
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    train_size = int(config.train_split * num_samples)
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_ds = Subset(train_source, train_indices)
    val_ds = Subset(eval_source, val_indices)
    test_ds = ImageFolder(Path(config.data_root) / "test", transform=eval_transform)

    train_targets = [train_source.samples[index][1] for index in train_indices]
    class_sample_count = np.bincount(train_targets)
    class_weights = 1.0 / class_sample_count
    sample_weights = [class_weights[target] for target in train_targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return {
        "classes": train_source.classes,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
