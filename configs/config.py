"""
Central configuration for all experiments.
Modify paths and hyperparameters here.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    raw_data_dir: str = "./rsna_data"
    processed_dir: str = "./rsna_processed"
    image_dir: str = "./rsna_processed/images"
    train_csv: str = "./rsna_processed/train_split.csv"
    val_csv: str = "./rsna_processed/val_split.csv"
    test_csv: str = "./rsna_processed/test_split.csv"
    original_csv: str = "./rsna_data/stage_2_train_labels.csv"

    # Unlabeled data for domain-adaptive pretraining
    unlabeled_dir: str = "./nih_chestxray/images"

    image_size: int = 1024
    num_workers: int = 4
    pin_memory: bool = True

    # Train/val/test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Standard augmentations (applied to all images)
    horizontal_flip_p: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2

    # Class-aware augmentation settings
    positive_oversample_factor: int = 2
    rotation_limit: int = 10
    scale_range: Tuple[float, float] = (0.9, 1.1)
    mosaic_p: float = 0.3

    # Augmentation mode: "standard", "class_aware", "none"
    mode: str = "standard"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Backbone
    backbone: str = "resnet50"
    pretrained_backbone: bool = True
    freeze_backbone_bn: bool = True

    # DETR-specific
    num_queries: int = 100
    hidden_dim: int = 256
    nheads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # Multi-scale DETR
    num_feature_levels: int = 4  # For multi-scale feature fusion
    num_deformable_points: int = 4

    # Detection
    num_classes: int = 2  # background + lung_opacity
    score_threshold: float = 0.5
    nms_iou_threshold: float = 0.5


@dataclass
class TrainConfig:
    """Training configuration."""
    # Optimizer
    optimizer: str = "adamw"
    lr_backbone: float = 1e-5
    lr_heads: float = 1e-4
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1

    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-7

    # Training
    epochs: int = 50
    batch_size: int = 4
    accumulation_steps: int = 2  # Effective batch = batch_size * accumulation_steps
    early_stopping_patience: int = 10

    # Loss weights (DETR Hungarian matching)
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0

    # Focal loss
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Checkpointing
    output_dir: str = "./outputs"
    save_every: int = 5
    log_every: int = 50
    use_wandb: bool = False
    project_name: str = "pneumonia_detection"


@dataclass
class MAEConfig:
    """Masked Autoencoder pretraining configuration."""
    mask_ratio: float = 0.75
    decoder_dim: int = 512
    decoder_depth: int = 8
    decoder_heads: int = 16
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1.5e-4
    warmup_epochs: int = 10
    image_size: int = 224  # Smaller for pretraining efficiency


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    iou_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    # Size bins for per-size AP (in pixels at 1024x1024)
    small_max: int = 96
    medium_max: int = 256
    # Visualization
    num_vis_samples: int = 50
    save_attention_maps: bool = True


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mae: MAEConfig = field(default_factory=MAEConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Experiment name (set via CLI)
    experiment_name: str = "default"
    seed: int = 42
    device: str = "cuda"

    def get_output_dir(self):
        path = os.path.join(self.train.output_dir, self.experiment_name)
        os.makedirs(path, exist_ok=True)
        return path


def get_config(**overrides) -> Config:
    """Create config with optional overrides."""
    cfg = Config()
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
    return cfg
