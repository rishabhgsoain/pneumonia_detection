"""
Evaluation metrics for object detection:
- mAP @ various IoU thresholds
- Per-size AP (small, medium, large)
- FROC analysis
- Precision-Recall curves
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou


def compute_iou_matrix(
    pred_boxes: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    """Compute IoU between all pairs of predicted and GT boxes."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.zeros(len(pred_boxes), len(gt_boxes))
    return box_iou(pred_boxes, gt_boxes)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_detections(
    all_predictions: List[Dict[str, torch.Tensor]],
    all_targets: List[Dict[str, torch.Tensor]],
    iou_thresholds: List[float] = None,
    size_ranges: Dict[str, Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive detection metrics.

    Args:
        all_predictions: list of dicts with 'boxes' [N,4], 'scores' [N], 'labels' [N]
        all_targets: list of dicts with 'boxes' [M,4], 'labels' [M]
        iou_thresholds: IoU thresholds for mAP (default: 0.5 to 0.95)
        size_ranges: dict mapping size name to (min_area, max_area)

    Returns:
        dict of metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    if size_ranges is None:
        size_ranges = {
            "small": (0, 96 ** 2),
            "medium": (96 ** 2, 256 ** 2),
            "large": (256 ** 2, float("inf")),
        }

    results = {}

    # Compute AP at each IoU threshold
    ap_per_threshold = []
    for iou_thr in iou_thresholds:
        ap = _compute_ap_at_threshold(all_predictions, all_targets, iou_thr)
        ap_per_threshold.append(ap)
        results[f"AP@{iou_thr:.2f}"] = ap

    results["mAP@0.5"] = results.get("AP@0.50", 0.0)
    results["mAP@0.5:0.95"] = np.mean(ap_per_threshold)

    # Per-size AP at IoU=0.5
    for size_name, (min_area, max_area) in size_ranges.items():
        ap = _compute_ap_at_threshold(
            all_predictions, all_targets, 0.5,
            min_area=min_area, max_area=max_area
        )
        results[f"AP@0.5_{size_name}"] = ap

    # Precision/Recall at IoU=0.5
    precision, recall, f1 = _compute_precision_recall(
        all_predictions, all_targets, iou_threshold=0.5
    )
    results["precision@0.5"] = precision
    results["recall@0.5"] = recall
    results["f1@0.5"] = f1

    # FROC-style: sensitivity at various FP/image rates
    froc_results = _compute_froc(all_predictions, all_targets)
    results.update(froc_results)

    return results


def _compute_ap_at_threshold(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float,
    min_area: float = 0,
    max_area: float = float("inf"),
) -> float:
    """Compute AP at a single IoU threshold, optionally filtered by box size."""
    all_scores = []
    all_tp = []
    n_gt = 0

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = tgt["boxes"]

        # Filter GT by size
        if len(gt_boxes) > 0:
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            size_mask = (gt_areas >= min_area) & (gt_areas < max_area)
            gt_boxes = gt_boxes[size_mask]

        n_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        # Sort by score descending
        sort_idx = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]

        if len(gt_boxes) == 0:
            all_scores.extend(pred_scores.cpu().numpy().tolist())
            all_tp.extend([0] * len(pred_scores))
            continue

        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = set()

        for i in range(len(pred_boxes)):
            all_scores.append(pred_scores[i].item())
            if iou_matrix.shape[1] > 0:
                max_iou, max_idx = iou_matrix[i].max(dim=0)
                max_idx = max_idx.item()
                if max_iou.item() >= iou_threshold and max_idx not in matched_gt:
                    all_tp.append(1)
                    matched_gt.add(max_idx)
                else:
                    all_tp.append(0)
            else:
                all_tp.append(0)

    if n_gt == 0:
        return 0.0

    # Sort by score
    sorted_indices = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[sorted_indices]
    fp = 1 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    return compute_ap(recalls, precisions)


def _compute_precision_recall(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 at given thresholds."""
    tp = 0
    fp = 0
    fn = 0

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = tgt["boxes"]

        # Filter by score
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]

        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue

        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue

        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = set()

        for i in range(len(pred_boxes)):
            if iou_matrix.shape[1] > 0:
                max_iou, max_idx = iou_matrix[i].max(dim=0)
                if max_iou.item() >= iou_threshold and max_idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(max_idx.item())
                else:
                    fp += 1
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def _compute_froc(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    fp_rates: List[float] = None,
) -> Dict[str, float]:
    """
    Compute FROC (Free-Response ROC) metrics.
    Reports sensitivity at various false positive per image rates.
    """
    if fp_rates is None:
        fp_rates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    n_images = len(predictions)
    all_scores = []
    all_tp = []
    n_gt = sum(len(t["boxes"]) for t in targets)

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = tgt["boxes"]

        if len(pred_boxes) == 0:
            continue

        sort_idx = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]

        if len(gt_boxes) == 0:
            all_scores.extend(pred_scores.cpu().numpy().tolist())
            all_tp.extend([0] * len(pred_scores))
            continue

        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = set()

        for i in range(len(pred_boxes)):
            all_scores.append(pred_scores[i].item())
            max_iou, max_idx = iou_matrix[i].max(dim=0)
            if max_iou.item() >= iou_threshold and max_idx.item() not in matched_gt:
                all_tp.append(1)
                matched_gt.add(max_idx.item())
            else:
                all_tp.append(0)

    if n_gt == 0 or len(all_scores) == 0:
        return {f"FROC_sens@{fp:.2f}fp": 0.0 for fp in fp_rates}

    sorted_idx = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[sorted_idx]
    fp = 1 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    sensitivity = tp_cumsum / n_gt
    fp_per_image = fp_cumsum / n_images

    results = {}
    for target_fp in fp_rates:
        idx = np.where(fp_per_image <= target_fp)[0]
        if len(idx) > 0:
            results[f"FROC_sens@{target_fp:.2f}fp"] = sensitivity[idx[-1]]
        else:
            results[f"FROC_sens@{target_fp:.2f}fp"] = 0.0

    return results
