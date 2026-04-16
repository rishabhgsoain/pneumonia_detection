"""
Training and evaluation engine.
Handles the training loop, validation, checkpointing, and logging.
"""
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from utils.metrics import evaluate_detections


class Trainer:
    """Unified trainer for all detection models."""

    def __init__(
        self,
        model: nn.Module,
        criterion,
        optimizer: torch.optim.Optimizer,
        scheduler,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.output_dir = cfg.get_output_dir()
        self.best_metric = 0.0
        self.patience_counter = 0

    def train(self) -> Dict:
        """Full training loop with validation and early stopping."""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.cfg.experiment_name}")
        print(f"{'='*60}")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Epochs: {self.cfg.train.epochs}")
        print(f"  Batch size: {self.cfg.train.batch_size}")
        print(f"  LR backbone: {self.cfg.train.lr_backbone}")
        print(f"  LR heads: {self.cfg.train.lr_heads}")
        print(f"  Output: {self.output_dir}")
        print()

        history = {"train_loss": [], "val_metrics": []}

        for epoch in range(self.cfg.train.epochs):
            # Train
            train_loss = self._train_one_epoch(epoch)
            history["train_loss"].append(train_loss)

            # Validate
            val_metrics = self._validate(epoch)
            history["val_metrics"].append(val_metrics)

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Checkpointing
            current_metric = val_metrics.get("mAP@0.5", 0.0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self._save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
                print(f"  >> New best mAP@0.5: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1

            if epoch % self.cfg.train.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.patience_counter >= self.cfg.train.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {self.patience_counter} epochs)")
                break

        print(f"\nTraining complete. Best mAP@0.5: {self.best_metric:.4f}")
        return history

    def _train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        accum_steps = self.cfg.train.accumulation_steps

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        self.optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            # Forward pass depends on model type
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'rpn'):
                # Faster R-CNN: returns losses directly
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
            else:
                # DETR: compute loss externally
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["total_loss"]

            # Gradient accumulation
            loss = loss / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                if self.cfg.train.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.clip_max_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            n_batches += 1

            if batch_idx % self.cfg.train.log_every == 0:
                avg_loss = total_loss / n_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        all_predictions = []
        all_targets = []

        for images, targets in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)

            # Get predictions
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'rpn'):
                # Faster R-CNN
                predictions = self.model(images)
            elif hasattr(self.model, 'predict'):
                # DETR models
                predictions = self.model.predict(
                    images, score_threshold=self.cfg.model.score_threshold
                )
            else:
                outputs = self.model(images)
                # Manual decoding
                predictions = _decode_detr_outputs(
                    outputs, images.shape, self.cfg.model.score_threshold
                )

            for pred in predictions:
                all_predictions.append({
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in pred.items()
                })
            for tgt in targets:
                all_targets.append({
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in tgt.items()
                })

        # Compute metrics
        metrics = evaluate_detections(
            all_predictions, all_targets,
            iou_thresholds=self.cfg.eval.iou_thresholds,
        )

        print(f"  Validation results (epoch {epoch+1}):")
        print(f"    mAP@0.5:     {metrics['mAP@0.5']:.4f}")
        print(f"    mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        print(f"    Precision:    {metrics['precision@0.5']:.4f}")
        print(f"    Recall:       {metrics['recall@0.5']:.4f}")
        print(f"    F1:           {metrics['f1@0.5']:.4f}")

        return metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.cfg,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.output_dir, "best_model.pth")
        else:
            path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pth")

        torch.save(checkpoint, path)


def _decode_detr_outputs(outputs, image_shape, score_threshold):
    """Decode DETR outputs into prediction dicts."""
    from torchvision.ops import box_convert
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    B = pred_logits.shape[0]
    results = []
    for i in range(B):
        probs = pred_logits[i].softmax(-1)
        scores = probs[:, 1]
        keep = scores > score_threshold
        boxes_cxcywh = pred_boxes[i][keep]
        boxes_xyxy = box_convert(boxes_cxcywh, "cxcywh", "xyxy")
        boxes_xyxy[:, [0, 2]] *= image_shape[3]
        boxes_xyxy[:, [1, 3]] *= image_shape[2]
        results.append({
            "boxes": boxes_xyxy,
            "scores": scores[keep],
            "labels": torch.ones(keep.sum(), dtype=torch.long),
        })
    return results
