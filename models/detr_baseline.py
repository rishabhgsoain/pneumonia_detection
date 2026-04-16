"""
Baseline Model: Vanilla DETR (DEtection TRansformer).
Implements the standard DETR architecture from Carion et al., ECCV 2020.
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import box_convert, generalized_box_iou

from configs.config import ModelConfig


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for spatial feature maps."""

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> pos: [B, C, H, W]"""
        B, C, H, W = x.shape
        assert C == self.d_model

        half_d = self.d_model // 2
        y_embed = torch.arange(H, device=x.device).float().unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=x.device).float().unsqueeze(0).expand(H, W)

        # Normalize to [0, 1]
        y_embed = y_embed / (H + 1e-6) * 2 * math.pi
        x_embed = x_embed / (W + 1e-6) * 2 * math.pi

        dim_t = torch.arange(half_d // 2, device=x.device).float()
        dim_t = self.temperature ** (2 * dim_t / half_d)

        pos_x = x_embed.unsqueeze(-1) / dim_t  # [H, W, D/4]
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)  # [H, W, D/2]
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_x, pos_y], dim=-1)  # [H, W, D]
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, D, H, W]

        return pos


class DETRBackbone(nn.Module):
    """ResNet-50 backbone with projection to hidden_dim."""

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)

        # Remove avgpool and fc
        self.body = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

        # Project from 2048 -> hidden_dim
        self.proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.body(x)  # [B, 2048, H/32, W/32]
        features = self.proj(features)  # [B, hidden_dim, H/32, W/32]
        return features


class DETR(nn.Module):
    """
    Standard DETR model.

    Architecture:
        1. CNN backbone extracts features
        2. Transformer encoder processes spatial features
        3. Transformer decoder attends to encoder output with learnable queries
        4. FFN heads predict class and bounding box for each query
    """

    def __init__(self, cfg: ModelConfig, pretrained_backbone: bool = True):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.num_queries = cfg.num_queries
        self.hidden_dim = cfg.hidden_dim

        # Backbone
        self.backbone = DETRBackbone(cfg.hidden_dim, pretrained_backbone)

        # Positional encoding
        self.pos_encoder = PositionalEncoding2D(cfg.hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.num_decoder_layers
        )

        # Learnable object queries
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)

        # Prediction heads
        self.class_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 4),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            dict with:
                pred_logits: [B, num_queries, num_classes]
                pred_boxes: [B, num_queries, 4] in cxcywh format, normalized
        """
        B = images.shape[0]

        # Backbone features
        features = self.backbone(images)  # [B, D, H', W']
        pos = self.pos_encoder(features)  # [B, D, H', W']

        # Flatten spatial dims for transformer
        H, W = features.shape[2], features.shape[3]
        src = features.flatten(2).permute(0, 2, 1)  # [B, H'*W', D]
        pos_flat = pos.flatten(2).permute(0, 2, 1)  # [B, H'*W', D]

        # Encoder
        memory = self.encoder(src + pos_flat)  # [B, H'*W', D]

        # Decoder
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        hs = self.decoder(query, memory)  # [B, Q, D]

        # Prediction heads
        pred_logits = self.class_head(hs)  # [B, Q, num_classes]
        pred_boxes = self.bbox_head(hs).sigmoid()  # [B, Q, 4] normalized cxcywh

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, score_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference and return filtered predictions.

        Returns:
            List of dicts per image with 'boxes' (xyxy), 'scores', 'labels'
        """
        outputs = self.forward(images)
        pred_logits = outputs["pred_logits"]  # [B, Q, C]
        pred_boxes = outputs["pred_boxes"]  # [B, Q, 4] cxcywh normalized

        B = pred_logits.shape[0]
        results = []

        for i in range(B):
            probs = pred_logits[i].softmax(-1)  # [Q, C]
            scores, labels = probs[:, :-1].max(-1) if self.num_classes > 2 else (
                probs[:, 1], torch.ones(probs.shape[0], device=probs.device, dtype=torch.long)
            )

            # Filter by score
            keep = scores > score_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes_cxcywh = pred_boxes[i][keep]

            # Convert to xyxy and scale to image size
            boxes_xyxy = box_convert(boxes_cxcywh, "cxcywh", "xyxy")
            # Scale to pixel coordinates
            img_h, img_w = images.shape[2], images.shape[3]
            boxes_xyxy[:, [0, 2]] *= img_w
            boxes_xyxy[:, [1, 3]] *= img_h

            results.append({
                "boxes": boxes_xyxy,
                "scores": scores,
                "labels": labels,
            })

        return results


def build_detr(cfg: ModelConfig, pretrained: bool = True) -> DETR:
    """Build vanilla DETR model."""
    return DETR(cfg, pretrained_backbone=pretrained)
