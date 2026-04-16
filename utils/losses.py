"""
Loss functions for object detection:
1. Focal Loss (for classification head)
2. Hungarian Matching Loss (for DETR-style set prediction)
3. GIoU Loss (for bounding box regression)
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert


class FocalLoss(nn.Module):
    """
    Focal Loss from Lin et al., ICCV 2017.
    Down-weights easy examples to focus on hard negatives.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] raw logits
            targets: [N] integer class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()


class HungarianMatcher(nn.Module):
    """
    Bipartite matching between predictions and ground truth.
    Finds optimal 1-to-1 assignment minimizing total cost.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: dict with pred_logits [B, Q, C] and pred_boxes [B, Q, 4]
            targets: list of dicts with boxes [N, 4] (xyxy) and labels [N]

        Returns:
            List of (pred_indices, target_indices) for each image
        """
        B, Q, C = outputs["pred_logits"].shape

        # Flatten predictions
        pred_logits = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, C]
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 4]

        # Concatenate targets
        tgt_labels = torch.cat([t["labels"] for t in targets])
        tgt_boxes = torch.cat([t["boxes"] for t in targets])

        if len(tgt_labels) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
                    for _ in range(B)]

        # Normalize target boxes to [0, 1] cxcywh if needed
        # Assuming targets are in xyxy pixel coords, normalize
        # (This is handled in the loss function; matcher works with whatever format)

        # Classification cost: -prob of correct class
        cost_class = -pred_logits[:, tgt_labels]  # [B*Q, N_targets]

        # L1 bbox cost
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)  # [B*Q, N_targets]

        # GIoU cost
        pred_boxes_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy")
        tgt_boxes_xyxy = box_convert(tgt_boxes, "cxcywh", "xyxy") if tgt_boxes.shape[-1] == 4 else tgt_boxes
        try:
            cost_giou = -generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)
        except Exception:
            cost_giou = torch.zeros_like(cost_bbox)

        # Total cost
        cost = (
            self.cost_class * cost_class +
            self.cost_bbox * cost_bbox +
            self.cost_giou * cost_giou
        )

        # Reshape to per-image costs and solve assignment
        cost = cost.view(B, Q, -1).cpu()

        sizes = [len(t["labels"]) for t in targets]
        indices = []
        offset = 0
        for i, s in enumerate(sizes):
            if s == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                ))
            else:
                c = cost[i, :, offset:offset + s]
                pred_idx, tgt_idx = linear_sum_assignment(c.numpy())
                indices.append((
                    torch.as_tensor(pred_idx, dtype=torch.long),
                    torch.as_tensor(tgt_idx, dtype=torch.long),
                ))
            offset += s

        return indices


class DETRLoss(nn.Module):
    """
    Full DETR loss combining classification, L1 box, and GIoU losses
    with Hungarian matching.
    """

    def __init__(
        self,
        num_classes: int = 2,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        eos_coef: float = 0.1,  # Weight for no-object class
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.eos_coef = eos_coef

        if use_focal:
            self.cls_loss_fn = FocalLoss(focal_alpha, focal_gamma)
        else:
            # Weighted CE with lower weight for no-object
            weight = torch.ones(num_classes)
            weight[-1] = eos_coef  # Assuming last class is "no object"
            self.register_buffer("class_weight", weight)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DETR loss.

        Args:
            outputs: pred_logits [B, Q, C], pred_boxes [B, Q, 4]
            targets: list of dicts with boxes and labels

        Returns:
            dict of losses: loss_ce, loss_bbox, loss_giou, total_loss
        """
        # Normalize target boxes to [0, 1] cxcywh for matching
        normalized_targets = []
        for t in targets:
            nt = {k: v for k, v in t.items()}
            if len(t["boxes"]) > 0:
                boxes = t["boxes"].clone()
                # Convert xyxy to cxcywh normalized (assuming 1024x1024)
                boxes_cxcywh = box_convert(boxes, "xyxy", "cxcywh")
                boxes_cxcywh[:, [0, 2]] /= 1024.0
                boxes_cxcywh[:, [1, 3]] /= 1024.0
                boxes_cxcywh = boxes_cxcywh.clamp(0, 1)
                nt["boxes"] = boxes_cxcywh
            normalized_targets.append(nt)

        # Hungarian matching
        indices = self.matcher(outputs, normalized_targets)

        # Classification loss
        B, Q, C = outputs["pred_logits"].shape
        device = outputs["pred_logits"].device

        # Create target labels (default: no-object class = num_classes - 1 or 0)
        target_classes = torch.zeros(B, Q, dtype=torch.long, device=device)
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[i, pred_idx] = normalized_targets[i]["labels"][tgt_idx].to(device)

        if self.use_focal:
            loss_ce = self.cls_loss_fn(
                outputs["pred_logits"].reshape(-1, C),
                target_classes.reshape(-1),
            )
        else:
            loss_ce = F.cross_entropy(
                outputs["pred_logits"].reshape(-1, C),
                target_classes.reshape(-1),
                weight=self.class_weight.to(device),
            )

        # Box losses (only for matched predictions)
        loss_bbox = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        n_boxes = 0

        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            pred_boxes = outputs["pred_boxes"][i, pred_idx]
            tgt_boxes = normalized_targets[i]["boxes"][tgt_idx].to(device)

            # L1 loss
            loss_bbox = loss_bbox + F.l1_loss(pred_boxes, tgt_boxes, reduction="sum")

            # GIoU loss
            pred_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy")
            tgt_xyxy = box_convert(tgt_boxes, "cxcywh", "xyxy")
            giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
            loss_giou = loss_giou + (1 - giou.diag()).sum()

            n_boxes += len(pred_idx)

        if n_boxes > 0:
            loss_bbox = loss_bbox / n_boxes
            loss_giou = loss_giou / n_boxes

        total_loss = (
            self.cost_class * loss_ce +
            self.cost_bbox * loss_bbox +
            self.cost_giou * loss_giou
        )

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "total_loss": total_loss,
        }
