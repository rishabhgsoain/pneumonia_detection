"""
Data augmentation pipelines for pneumonia detection.
Includes standard, class-aware, and mosaic augmentation strategies.
"""
import random
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np


def get_standard_augmentation(image_size: int = 1024) -> A.Compose:
    """Standard augmentation for all models."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


def get_class_aware_augmentation(image_size: int = 1024) -> A.Compose:
    """
    Aggressive augmentation for positive (pneumonia) images.
    Standard augmentation is used for negative images via the dataset class.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0,
                    ),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                ],
                p=0.3,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


def get_validation_augmentation(image_size: int = 1024) -> A.Compose:
    """No augmentation for validation/test — just pass through."""
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


class MosaicAugmentation:
    """
    Mosaic augmentation: combine 4 images into one.
    Useful for exposing the model to multiple objects at different scales.
    """

    def __init__(self, image_size: int = 1024, p: float = 0.3):
        self.image_size = image_size
        self.p = p

    def __call__(
        self,
        images: List[np.ndarray],
        all_boxes: List[np.ndarray],
        all_labels: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            images: list of 4 images (H, W, C)
            all_boxes: list of 4 box arrays, each (N, 4) in pascal_voc
            all_labels: list of 4 label arrays

        Returns:
            mosaic_image, mosaic_boxes, mosaic_labels
        """
        if random.random() > self.p or len(images) < 4:
            return images[0], all_boxes[0], all_labels[0]

        s = self.image_size
        # Random center point
        cx = random.randint(s // 4, 3 * s // 4)
        cy = random.randint(s // 4, 3 * s // 4)

        mosaic = np.zeros((s, s, 3), dtype=np.uint8)
        final_boxes = []
        final_labels = []

        # Placement regions: top-left, top-right, bottom-left, bottom-right
        placements = [
            (0, 0, cx, cy),
            (cx, 0, s, cy),
            (0, cy, cx, s),
            (cx, cy, s, s),
        ]

        for i, (x1, y1, x2, y2) in enumerate(placements):
            img = images[i]
            h, w = img.shape[:2]

            # Resize to fit placement region
            rw, rh = x2 - x1, y2 - y1
            if rw <= 0 or rh <= 0:
                continue

            scale_x = rw / w
            scale_y = rh / h
            resized = cv2.resize(img, (rw, rh))
            mosaic[y1:y2, x1:x2] = resized

            # Adjust boxes
            boxes = all_boxes[i].copy() if len(all_boxes[i]) > 0 else np.zeros((0, 4))
            if len(boxes) > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + x1
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + y1

                # Clip to mosaic bounds
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], x1, x2)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], y1, y2)

                # Filter tiny boxes
                w_box = boxes[:, 2] - boxes[:, 0]
                h_box = boxes[:, 3] - boxes[:, 1]
                valid = (w_box > 10) & (h_box > 10)
                boxes = boxes[valid]

                if len(boxes) > 0:
                    final_boxes.append(boxes)
                    final_labels.append(all_labels[i][valid])

        if final_boxes:
            final_boxes = np.concatenate(final_boxes, axis=0)
            final_labels = np.concatenate(final_labels, axis=0)
        else:
            final_boxes = np.zeros((0, 4), dtype=np.float32)
            final_labels = np.zeros(0, dtype=np.int64)

        return mosaic, final_boxes, final_labels


def build_augmentation(mode: str, image_size: int = 1024):
    """Factory function to build augmentation pipeline."""
    if mode == "none":
        return get_validation_augmentation(image_size)
    elif mode == "standard":
        return get_standard_augmentation(image_size)
    elif mode == "class_aware":
        return get_class_aware_augmentation(image_size)
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")
