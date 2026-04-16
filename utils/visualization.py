"""
Visualization utilities for analysis and reporting.
- Bounding box overlays
- Attention map visualization
- Error analysis plots
- Metric comparison charts
"""
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# Consistent color scheme
COLORS = {
    "gt": "#00FF00",       # Green for ground truth
    "pred_tp": "#0088FF",  # Blue for true positive
    "pred_fp": "#FF0000",  # Red for false positive
    "missed": "#FFAA00",   # Orange for missed GT
}


def visualize_predictions(
    image: np.ndarray,
    predictions: Dict,
    targets: Dict,
    iou_threshold: float = 0.5,
    save_path: Optional[str] = None,
    title: str = "",
):
    """
    Visualize predicted and ground truth bounding boxes on an image.
    Color codes: TP (blue), FP (red), GT (green), Missed (orange).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

    gt_boxes = targets["boxes"].numpy() if isinstance(targets["boxes"], torch.Tensor) else targets["boxes"]
    pred_boxes = predictions["boxes"].numpy() if isinstance(predictions["boxes"], torch.Tensor) else predictions["boxes"]
    pred_scores = predictions["scores"].numpy() if isinstance(predictions["scores"], torch.Tensor) else predictions["scores"]

    # Match predictions to GT
    matched_gt = set()
    pred_status = []  # "tp" or "fp" for each prediction

    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        from utils.metrics import compute_iou_matrix
        iou_matrix = compute_iou_matrix(
            torch.tensor(pred_boxes), torch.tensor(gt_boxes)
        ).numpy()

        sort_idx = np.argsort(-pred_scores)
        for i in sort_idx:
            if iou_matrix.shape[1] > 0:
                max_iou_idx = iou_matrix[i].argmax()
                if iou_matrix[i, max_iou_idx] >= iou_threshold and max_iou_idx not in matched_gt:
                    pred_status.append("tp")
                    matched_gt.add(max_iou_idx)
                else:
                    pred_status.append("fp")
            else:
                pred_status.append("fp")
        # Reorder to original order
        reorder = np.argsort(sort_idx)
        pred_status = [pred_status[r] for r in reorder]
    else:
        pred_status = ["fp"] * len(pred_boxes)

    # Draw GT boxes
    for i, box in enumerate(gt_boxes):
        color = COLORS["missed"] if i not in matched_gt else COLORS["gt"]
        label = "Missed GT" if i not in matched_gt else "GT"
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)

    # Draw predictions
    for i, (box, score, status) in enumerate(zip(pred_boxes, pred_scores, pred_status)):
        color = COLORS[f"pred_{status}"]
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            box[0], box[1] - 5,
            f"{score:.2f} ({status.upper()})",
            color=color, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7)
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["gt"], linewidth=2, linestyle="--", label="Ground Truth"),
        Line2D([0], [0], color=COLORS["pred_tp"], linewidth=2, label="True Positive"),
        Line2D([0], [0], color=COLORS["pred_fp"], linewidth=2, label="False Positive"),
        Line2D([0], [0], color=COLORS["missed"], linewidth=2, linestyle="--", label="Missed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_attention_maps(
    model,
    images: torch.Tensor,
    save_dir: str,
    num_samples: int = 10,
):
    """
    Extract and visualize attention maps from transformer encoder.
    Shows which image regions the model attends to.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Hook to capture attention weights
    attention_maps = []

    def hook_fn(module, input, output):
        # TransformerEncoderLayer stores self-attention in multihead_attn
        if hasattr(module, "self_attn"):
            attention_maps.append(output)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.TransformerEncoderLayer):
            hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        for i in range(min(num_samples, len(images))):
            attention_maps.clear()
            image = images[i:i+1]
            _ = model(image)

            if attention_maps:
                # Average attention across heads and layers
                attn = attention_maps[-1]
                if isinstance(attn, torch.Tensor):
                    attn_map = attn[0].mean(0)  # Average over sequence
                    h = int(attn_map.shape[0] ** 0.5)
                    if h * h == attn_map.shape[0]:
                        attn_map = attn_map.view(h, h)
                        attn_map = F.interpolate(
                            attn_map.unsqueeze(0).unsqueeze(0),
                            size=(images.shape[2], images.shape[3]),
                            mode="bilinear",
                        )[0, 0]

                        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                        img_np = _denormalize(image[0].cpu())
                        axes[0].imshow(img_np)
                        axes[0].set_title("Input Image")
                        axes[0].axis("off")

                        axes[1].imshow(img_np)
                        axes[1].imshow(attn_map.cpu().numpy(), alpha=0.5, cmap="jet")
                        axes[1].set_title("Attention Overlay")
                        axes[1].axis("off")

                        plt.savefig(
                            os.path.join(save_dir, f"attention_{i}.png"),
                            dpi=150, bbox_inches="tight"
                        )
                        plt.close()

    for h in hooks:
        h.remove()


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
):
    """
    Bar chart comparing metrics across all experiments.

    Args:
        results: {experiment_name: {metric_name: value}}
    """
    experiments = list(results.keys())
    metrics = ["mAP@0.5", "mAP@0.5:0.95", "precision@0.5", "recall@0.5", "f1@0.5"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    for j, metric in enumerate(metrics):
        values = [results[exp].get(metric, 0) for exp in experiments]
        bars = axes[j].bar(range(len(experiments)), values, color=colors)
        axes[j].set_title(metric, fontsize=11, fontweight="bold")
        axes[j].set_xticks(range(len(experiments)))
        axes[j].set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
        axes[j].set_ylim(0, 1)
        axes[j].grid(axis="y", alpha=0.3)

        # Value labels
        for bar, val in zip(bars, values):
            axes[j].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8
            )

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_analysis(
    all_predictions: List[Dict],
    all_targets: List[Dict],
    save_dir: str,
):
    """
    Generate error analysis plots:
    1. Distribution of FP/FN by box size
    2. Confidence score distribution for TP vs FP
    3. IoU distribution of matched predictions
    """
    os.makedirs(save_dir, exist_ok=True)
    from utils.metrics import compute_iou_matrix

    tp_scores, fp_scores = [], []
    tp_ious = []
    fn_sizes, fp_sizes = [], []

    for pred, tgt in zip(all_predictions, all_targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = tgt["boxes"]

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
            matched_gt = set()

            sort_idx = pred_scores.argsort(descending=True)
            for i in sort_idx:
                max_iou, max_idx = iou_matrix[i].max(dim=0)
                if max_iou.item() >= 0.5 and max_idx.item() not in matched_gt:
                    tp_scores.append(pred_scores[i].item())
                    tp_ious.append(max_iou.item())
                    matched_gt.add(max_idx.item())
                else:
                    fp_scores.append(pred_scores[i].item())
                    area = (pred_boxes[i, 2] - pred_boxes[i, 0]) * (pred_boxes[i, 3] - pred_boxes[i, 1])
                    fp_sizes.append(area.item())

            for j in range(len(gt_boxes)):
                if j not in matched_gt:
                    area = (gt_boxes[j, 2] - gt_boxes[j, 0]) * (gt_boxes[j, 3] - gt_boxes[j, 1])
                    fn_sizes.append(area.item())
        elif len(pred_boxes) > 0:
            fp_scores.extend(pred_scores.numpy().tolist())
        elif len(gt_boxes) > 0:
            for box in gt_boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                fn_sizes.append(area.item())

    # Plot 1: Score distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(tp_scores, bins=30, alpha=0.7, label=f"TP (n={len(tp_scores)})", color="blue")
    axes[0].hist(fp_scores, bins=30, alpha=0.7, label=f"FP (n={len(fp_scores)})", color="red")
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution: TP vs FP")
    axes[0].legend()

    # Plot 2: IoU distribution
    if tp_ious:
        axes[1].hist(tp_ious, bins=30, color="green", alpha=0.7)
    axes[1].set_xlabel("IoU with Ground Truth")
    axes[1].set_ylabel("Count")
    axes[1].set_title("IoU Distribution of True Positives")

    # Plot 3: Size distribution of errors
    if fn_sizes:
        axes[2].hist(np.sqrt(fn_sizes), bins=30, alpha=0.7, label=f"Missed GT (n={len(fn_sizes)})", color="orange")
    if fp_sizes:
        axes[2].hist(np.sqrt(fp_sizes), bins=30, alpha=0.7, label=f"False Pos (n={len(fp_sizes)})", color="red")
    axes[2].set_xlabel("Box Size (sqrt area, pixels)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Error Distribution by Box Size")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img
