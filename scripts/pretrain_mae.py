"""
Domain-adaptive MAE pretraining on unlabeled chest X-rays.

Usage:
    python scripts/pretrain_mae.py --image-dir ./nih_chestxray/images --epochs 100
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import get_config
from data.dataset import UnlabeledChestXrayDataset
from models.domain_pretrain import MaskedAutoencoder, pretrain_mae


def main():
    parser = argparse.ArgumentParser(description="MAE pretraining on chest X-rays")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory with unlabeled chest X-ray images")
    parser.add_argument("--output-path", type=str,
                        default="./outputs/mae_pretrained.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build dataset
    dataset = UnlabeledChestXrayDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Build MAE model
    model = MaskedAutoencoder(
        hidden_dim=256,
        decoder_dim=512,
        decoder_depth=4,
        decoder_heads=8,
        mask_ratio=args.mask_ratio,
        image_size=args.image_size,
        pretrained_backbone=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Pretrain
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    pretrain_mae(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_path=args.output_path,
    )


if __name__ == "__main__":
    main()
