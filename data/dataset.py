"""
RSNA Pneumonia Detection dataset for PyTorch.
Supports both Faster R-CNN and DETR format outputs.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from configs.config import Config


class RSNAPneumoniaDataset(Dataset):
    """
    RSNA Pneumonia Detection dataset.

    Returns images and targets in COCO-like format compatible with
    both torchvision Faster R-CNN and DETR.

    Target format:
        - boxes: [N, 4] in (x1, y1, x2, y2) format, normalized to [0, 1]
        - labels: [N] integer class labels (1 = lung_opacity)
        - image_id: unique image identifier
        - area: [N] box areas
        - iscrowd: [N] always 0
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform=None,
        image_size: int = 1024,
        is_train: bool = True,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.is_train = is_train

        # Load annotations
        df = pd.read_csv(csv_path)

        # Group by patient: each patient = one image
        self.patient_ids = df["patientId"].unique().tolist()

        # Build annotation lookup: patientId -> list of boxes
        self.annotations = {}
        for pid in self.patient_ids:
            patient_rows = df[df["patientId"] == pid]
            boxes = []
            for _, row in patient_rows.iterrows():
                if row["Target"] == 1 and pd.notna(row["x"]):
                    # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                    x1 = float(row["x"])
                    y1 = float(row["y"])
                    x2 = x1 + float(row["w"])
                    y2 = y1 + float(row["h"])
                    boxes.append([x1, y1, x2, y2])
            self.annotations[pid] = boxes

        # Track which images are positive (for class-aware sampling)
        self.positive_indices = [
            i for i, pid in enumerate(self.patient_ids)
            if len(self.annotations[pid]) > 0
        ]
        self.negative_indices = [
            i for i, pid in enumerate(self.patient_ids)
            if len(self.annotations[pid]) == 0
        ]

        print(f"  Loaded {len(self.patient_ids)} images "
              f"({len(self.positive_indices)} positive, "
              f"{len(self.negative_indices)} negative)")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        pid = self.patient_ids[idx]

        # Load image
        img_path = os.path.join(self.image_dir, f"{pid}.png")
        image = Image.open(img_path).convert("RGB")

        # Get boxes
        boxes = self.annotations[pid]
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones(len(boxes), dtype=torch.int64)  # class 1 = pneumonia
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            areas = torch.zeros(0, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "patient_id": pid,
        }

        # Apply augmentations (albumentations-based)
        if self.transform is not None:
            image, target = self._apply_transform(image, target)
        else:
            image = self._default_transform(image)

        return image, target

    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Default: resize and normalize."""
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image, dtype=np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _apply_transform(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """Apply albumentations transform."""
        image = image.resize((self.image_size, self.image_size))
        image_np = np.array(image)

        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        if len(boxes) > 0:
            # Albumentations expects pascal_voc format: [x_min, y_min, x_max, y_max]
            transformed = self.transform(
                image=image_np,
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
            )
        else:
            transformed = self.transform(image=image_np, bboxes=[], labels=[])

        image_np = transformed["image"]

        # Normalize
        image_np = image_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        if len(transformed["bboxes"]) > 0:
            target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
            target["area"] = (
                (target["boxes"][:, 2] - target["boxes"][:, 0]) *
                (target["boxes"][:, 3] - target["boxes"][:, 1])
            )
            target["iscrowd"] = torch.zeros(len(target["boxes"]), dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["area"] = torch.zeros(0, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(0, dtype=torch.int64)

        return image_tensor, target


class UnlabeledChestXrayDataset(Dataset):
    """Unlabeled chest X-ray dataset for domain-adaptive pretraining."""

    def __init__(self, image_dir: str, transform=None, image_size: int = 224):
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size

        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"  Loaded {len(self.image_files)} unlabeled images for pretraining")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image


def collate_fn(batch):
    """Custom collate for variable number of bounding boxes."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets


def build_dataloaders(cfg: Config, train_transform=None, val_transform=None):
    """Build train, val, and test dataloaders."""
    print("Building dataloaders...")

    train_dataset = RSNAPneumoniaDataset(
        csv_path=cfg.data.train_csv,
        image_dir=cfg.data.image_dir,
        transform=train_transform,
        image_size=cfg.data.image_size,
        is_train=True,
    )
    val_dataset = RSNAPneumoniaDataset(
        csv_path=cfg.data.val_csv,
        image_dir=cfg.data.image_dir,
        transform=val_transform,
        image_size=cfg.data.image_size,
        is_train=False,
    )
    test_dataset = RSNAPneumoniaDataset(
        csv_path=cfg.data.test_csv,
        image_dir=cfg.data.image_dir,
        transform=val_transform,
        image_size=cfg.data.image_size,
        is_train=False,
    )

    # Optional: weighted sampler for class-aware training
    sampler = None
    shuffle = True
    if cfg.augmentation.mode == "class_aware":
        weights = np.ones(len(train_dataset))
        for i in train_dataset.positive_indices:
            weights[i] = cfg.augmentation.positive_oversample_factor
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
