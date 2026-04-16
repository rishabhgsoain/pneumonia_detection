"""
Baseline Model: Faster R-CNN with ResNet-50-FPN backbone.
Uses torchvision's built-in implementation.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from configs.config import ModelConfig


def build_faster_rcnn(cfg: ModelConfig, pretrained: bool = True) -> nn.Module:
    """
    Build Faster R-CNN with ResNet-50-FPN backbone.

    Adaptations for chest X-ray pneumonia detection:
    - Larger anchor sizes (pneumonia boxes tend to be large)
    - 2 classes: background + lung_opacity
    - Lower NMS threshold for overlapping opacities
    """
    # Load pretrained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        min_size=800,
        max_size=1024,
    )

    # Custom anchor sizes for chest X-rays
    # Pneumonia boxes are typically medium to large
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    anchor_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=anchor_ratios,
    )
    model.rpn.anchor_generator = anchor_generator

    # Replace classification head for 2 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=cfg.num_classes
    )

    # Adjust NMS and score thresholds
    model.roi_heads.score_thresh = cfg.score_threshold
    model.roi_heads.nms_thresh = cfg.nms_iou_threshold

    return model


class FasterRCNNWrapper(nn.Module):
    """
    Wrapper to provide a consistent interface across all models.

    In training mode: returns loss dict
    In eval mode: returns list of prediction dicts
    """

    def __init__(self, cfg: ModelConfig, pretrained: bool = True):
        super().__init__()
        self.model = build_faster_rcnn(cfg, pretrained)

    def forward(self, images, targets=None):
        """
        Args:
            images: tensor [B, 3, H, W]
            targets: list of dicts with 'boxes' and 'labels' (training only)

        Returns:
            training: dict of losses
            eval: list of dicts with 'boxes', 'scores', 'labels'
        """
        if self.training and targets is not None:
            # torchvision Faster R-CNN expects a list of images
            image_list = [img for img in images]
            target_list = []
            for t in targets:
                target_list.append({
                    "boxes": t["boxes"],
                    "labels": t["labels"],
                })
            loss_dict = self.model(image_list, target_list)
            return loss_dict
        else:
            image_list = [img for img in images]
            predictions = self.model(image_list)
            return predictions
