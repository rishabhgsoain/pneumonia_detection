"""
Main training script for all experiments.

Usage:
    # E1: Faster R-CNN baseline
    python scripts/train.py --model faster_rcnn --experiment E1_faster_rcnn

    # E2: Vanilla DETR baseline
    python scripts/train.py --model detr --experiment E2_detr_vanilla

    # E3: DETR + domain pretraining
    python scripts/train.py --model detr --domain-pretrain --experiment E3_detr_pretrained

    # E4: Multi-scale DETR
    python scripts/train.py --model detr_multiscale --experiment E4_detr_multiscale

    # E5: DETR + class-aware augmentation
    python scripts/train.py --model detr --augment class_aware --experiment E5_detr_augmented

    # E6: Full pipeline
    python scripts/train.py --model detr_multiscale --domain-pretrain --augment class_aware --experiment E6_full_pipeline
"""
import argparse
import os
import sys
import json
import random

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import get_config
from data.dataset import build_dataloaders
from data.augmentations import build_augmentation
from utils.engine import Trainer
from utils.losses import DETRLoss


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name: str, cfg, pretrain_path=None):
    """Build the detection model."""
    if model_name == "faster_rcnn":
        from models.faster_rcnn import FasterRCNNWrapper
        model = FasterRCNNWrapper(cfg.model, pretrained=True)

    elif model_name == "detr":
        from models.detr_baseline import build_detr
        model = build_detr(cfg.model, pretrained=True)

        # Optionally load domain-pretrained backbone
        if pretrain_path and os.path.exists(pretrain_path):
            print(f"Loading domain-pretrained backbone from {pretrain_path}")
            state_dict = torch.load(pretrain_path, map_location="cpu")
            # Map MAE encoder keys to DETR backbone keys
            backbone_state = {}
            for k, v in state_dict.items():
                new_key = k.replace("features.", "body.").replace("proj.", "proj.")
                backbone_state[new_key] = v
            model.backbone.load_state_dict(backbone_state, strict=False)

    elif model_name == "detr_multiscale":
        from models.detr_multiscale import build_detr_multiscale
        model = build_detr_multiscale(cfg.model, pretrained=True)

        if pretrain_path and os.path.exists(pretrain_path):
            print(f"Loading domain-pretrained backbone from {pretrain_path}")
            state_dict = torch.load(pretrain_path, map_location="cpu")
            # Load into FPN backbone (matching layer names)
            model.backbone.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def build_optimizer(model, cfg, model_name):
    """Build optimizer with different LR for backbone and heads."""
    if model_name == "faster_rcnn":
        # Faster R-CNN: single LR group
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=cfg.train.lr_heads, weight_decay=cfg.train.weight_decay
        )
    else:
        # DETR: separate LR for backbone
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.train.lr_backbone},
            {"params": head_params, "lr": cfg.train.lr_heads},
        ], weight_decay=cfg.train.weight_decay)

    return optimizer


def build_scheduler(optimizer, cfg):
    """Build learning rate scheduler."""
    if cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs - cfg.train.warmup_epochs,
            eta_min=cfg.train.min_lr,
        )
    elif cfg.train.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None

    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["faster_rcnn", "detr", "detr_multiscale"])
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--augment", type=str, default="standard",
                        choices=["none", "standard", "class_aware"])
    parser.add_argument("--domain-pretrain", action="store_true")
    parser.add_argument("--pretrain-path", type=str,
                        default="./outputs/mae_pretrained.pth")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build config
    overrides = {"experiment_name": args.experiment, "seed": args.seed, "device": args.device}
    overrides["augmentation.mode"] = args.augment
    if args.epochs:
        overrides["train.epochs"] = args.epochs
    if args.batch_size:
        overrides["train.batch_size"] = args.batch_size
    if args.lr:
        overrides["train.lr_heads"] = args.lr

    cfg = get_config(**overrides)
    set_seed(cfg.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build augmentations
    train_transform = build_augmentation(args.augment, cfg.data.image_size)
    val_transform = build_augmentation("none", cfg.data.image_size)

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, train_transform=train_transform, val_transform=val_transform
    )

    # Build model
    pretrain_path = args.pretrain_path if args.domain_pretrain else None
    model = build_model(args.model, cfg, pretrain_path)

    # Build criterion
    if args.model == "faster_rcnn":
        criterion = None  # Faster R-CNN computes loss internally
    else:
        criterion = DETRLoss(
            num_classes=cfg.model.num_classes,
            cost_class=cfg.train.cost_class,
            cost_bbox=cfg.train.cost_bbox,
            cost_giou=cfg.train.cost_giou,
            use_focal=cfg.train.use_focal_loss,
            focal_alpha=cfg.train.focal_alpha,
            focal_gamma=cfg.train.focal_gamma,
        )

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg, args.model)
    scheduler = build_scheduler(optimizer, cfg)

    # Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    history = trainer.train()

    # Save training history
    output_dir = cfg.get_output_dir()
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump({
            "train_loss": history["train_loss"],
            "val_metrics": [{k: float(v) for k, v in m.items()} for m in history["val_metrics"]],
            "config": {
                "model": args.model,
                "augment": args.augment,
                "domain_pretrain": args.domain_pretrain,
                "epochs": cfg.train.epochs,
                "batch_size": cfg.train.batch_size,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
