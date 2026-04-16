"""
Proposed Model: Multi-Scale DETR with Feature Pyramid Network.

Key differences from vanilla DETR:
1. FPN backbone produces multi-scale features (1/8, 1/16, 1/32)
2. Deformable attention in encoder for efficient multi-scale processing
3. Scale-aware query initialization
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import box_convert

from configs.config import ModelConfig


class FPNBackbone(nn.Module):
    """
    ResNet-50 with Feature Pyramid Network producing multi-scale features.
    Outputs feature maps at strides 8, 16, and 32.
    """

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)

        # Extract intermediate layers
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # stride 4, 256 channels
        self.layer2 = backbone.layer2  # stride 8, 512 channels
        self.layer3 = backbone.layer3  # stride 16, 1024 channels
        self.layer4 = backbone.layer4  # stride 32, 2048 channels

        # Lateral connections (1x1 convs to reduce channel dims)
        self.lateral4 = nn.Conv2d(2048, hidden_dim, 1)
        self.lateral3 = nn.Conv2d(1024, hidden_dim, 1)
        self.lateral2 = nn.Conv2d(512, hidden_dim, 1)

        # Output convolutions (3x3 to smooth after addition)
        self.output4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.output3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.output2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        # Level embedding to distinguish feature scales
        self.level_embed = nn.Parameter(torch.randn(3, hidden_dim))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns list of feature maps at different scales.
        Each: [B, hidden_dim, H_i, W_i]
        """
        x = self.layer0(x)
        c2 = self.layer1(x)   # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        # Top-down pathway
        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        # Smooth
        p5 = self.output4(p5)
        p4 = self.output3(p4)
        p3 = self.output2(p3)

        # Add level embeddings
        B = x.shape[0]
        p3 = p3 + self.level_embed[0].view(1, -1, 1, 1)
        p4 = p4 + self.level_embed[1].view(1, -1, 1, 1)
        p5 = p5 + self.level_embed[2].view(1, -1, 1, 1)

        return [p3, p4, p5]  # stride 8, 16, 32


class MultiScalePositionalEncoding(nn.Module):
    """Learned positional encoding for multi-scale features."""

    def __init__(self, hidden_dim: int, max_size: int = 128):
        super().__init__()
        self.row_embed = nn.Embedding(max_size, hidden_dim // 2)
        self.col_embed = nn.Embedding(max_size, hidden_dim // 2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """feature: [B, C, H, W] -> pos: [B, C, H, W]"""
        H, W = feature.shape[2], feature.shape[3]
        rows = self.row_embed(torch.arange(H, device=feature.device))  # [H, D/2]
        cols = self.col_embed(torch.arange(W, device=feature.device))  # [W, D/2]

        pos = torch.cat([
            cols.unsqueeze(0).expand(H, -1, -1),
            rows.unsqueeze(1).expand(-1, W, -1),
        ], dim=-1)  # [H, W, D]

        pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
        return pos.expand(feature.shape[0], -1, -1, -1)


class DeformableAttention(nn.Module):
    """
    Simplified deformable attention for multi-scale features.
    Each query attends to a set of learned sampling points around a reference.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 3,
        n_points: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2
        )
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        query: torch.Tensor,
        multi_scale_values: List[torch.Tensor],
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified deformable attention via bilinear sampling.

        Args:
            query: [B, Q, D]
            multi_scale_values: list of [B, D, H_i, W_i]
            reference_points: [B, Q, 2] normalized (0-1) reference xy

        Returns:
            output: [B, Q, D]
        """
        B, Q, D = query.shape
        n_heads = self.n_heads
        n_points = self.n_points
        n_levels = len(multi_scale_values)
        head_dim = D // n_heads

        # Predict sampling offsets from query
        offsets = self.sampling_offsets(query)  # [B, Q, H*L*P*2]
        offsets = offsets.view(B, Q, n_heads, n_levels, n_points, 2)
        offsets = offsets.tanh() * 0.5  # Limit offset range

        # Predict attention weights
        attn_weights = self.attention_weights(query)  # [B, Q, H*L*P]
        attn_weights = attn_weights.view(B, Q, n_heads, n_levels * n_points)
        attn_weights = attn_weights.softmax(-1)
        attn_weights = attn_weights.view(B, Q, n_heads, n_levels, n_points)

        # Sample values from each level
        output = torch.zeros(B, Q, n_heads, head_dim, device=query.device)

        for lvl, feat_map in enumerate(multi_scale_values):
            value = self.value_proj(feat_map.flatten(2).permute(0, 2, 1))  # [B, HW, D]
            value = value.view(B, feat_map.shape[2], feat_map.shape[3], n_heads, head_dim)
            value = value.permute(0, 3, 4, 1, 2)  # [B, H_heads, head_dim, H_feat, W_feat]

            # Compute sampling locations
            ref = reference_points.unsqueeze(2).unsqueeze(4)  # [B, Q, 1, 1, 2]
            lvl_offsets = offsets[:, :, :, lvl, :, :]  # [B, Q, H_heads, P, 2]
            sampling_locs = ref + lvl_offsets  # [B, Q, H_heads, P, 2]
            # Convert to grid_sample format: [-1, 1]
            sampling_grid = sampling_locs * 2 - 1  # [B, Q, H_heads, P, 2]

            for h in range(n_heads):
                grid = sampling_grid[:, :, h, :, :]  # [B, Q, P, 2]
                grid = grid.view(B, Q, n_points, 2)
                # Pad to 4D for grid_sample: [B, head_dim, Q, P]
                val = value[:, h]  # [B, head_dim, H, W]
                sampled = F.grid_sample(
                    val, grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )  # [B, head_dim, Q, P]
                sampled = sampled.permute(0, 2, 3, 1)  # [B, Q, P, head_dim]
                w = attn_weights[:, :, h, lvl, :]  # [B, Q, P]
                output[:, :, h] += (sampled * w.unsqueeze(-1)).sum(dim=2)

        output = output.view(B, Q, D)
        output = self.output_proj(output)
        return output


class MultiScaleDETR(nn.Module):
    """
    Multi-Scale DETR with FPN backbone and deformable attention.

    Improvements over vanilla DETR:
    1. Multi-scale features via FPN (captures small and large pathologies)
    2. Deformable attention (efficient, focuses on relevant regions)
    3. Learned reference points for queries
    """

    def __init__(self, cfg: ModelConfig, pretrained_backbone: bool = True):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.num_queries = cfg.num_queries
        self.hidden_dim = cfg.hidden_dim

        # FPN backbone
        self.backbone = FPNBackbone(cfg.hidden_dim, pretrained_backbone)

        # Positional encoding (one per level)
        self.pos_encoders = nn.ModuleList([
            MultiScalePositionalEncoding(cfg.hidden_dim) for _ in range(3)
        ])

        # Encoder with deformable attention
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": DeformableAttention(
                    cfg.hidden_dim, cfg.nheads, n_levels=3, n_points=cfg.num_deformable_points
                ),
                "norm1": nn.LayerNorm(cfg.hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.dim_feedforward, cfg.hidden_dim),
                    nn.Dropout(cfg.dropout),
                ),
                "norm2": nn.LayerNorm(cfg.hidden_dim),
            })
            for _ in range(cfg.num_encoder_layers)
        ])

        # Decoder (standard cross-attention)
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

        # Object queries and reference points
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.reference_points = nn.Linear(cfg.hidden_dim, 2)

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
        # Initialize reference points to spread across image
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.5)

    def _encode_multi_scale(
        self, multi_scale_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Run deformable encoder over multi-scale features.
        Returns flattened encoded features.
        """
        B = multi_scale_features[0].shape[0]

        # Add positional encoding to each level
        for i, (feat, pos_enc) in enumerate(zip(multi_scale_features, self.pos_encoders)):
            multi_scale_features[i] = feat + pos_enc(feat)

        # Flatten all features into a single sequence
        flat_features = []
        spatial_shapes = []
        for feat in multi_scale_features:
            H, W = feat.shape[2], feat.shape[3]
            spatial_shapes.append((H, W))
            flat_features.append(feat.flatten(2).permute(0, 2, 1))  # [B, HW, D]

        tokens = torch.cat(flat_features, dim=1)  # [B, sum(HW), D]

        # Generate reference points for encoder tokens
        ref_points = []
        for H, W in spatial_shapes:
            ys = torch.linspace(0, 1, H, device=tokens.device)
            xs = torch.linspace(0, 1, W, device=tokens.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            ref = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            ref_points.append(ref)
        ref_points = torch.cat(ref_points, dim=0).unsqueeze(0).expand(B, -1, -1)

        # Run encoder layers
        for layer in self.encoder_layers:
            residual = tokens
            tokens = layer["norm1"](
                residual + layer["self_attn"](tokens, multi_scale_features, ref_points)
            )
            residual = tokens
            tokens = layer["norm2"](residual + layer["ffn"](tokens))

        return tokens

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            pred_logits: [B, num_queries, num_classes]
            pred_boxes: [B, num_queries, 4] in cxcywh, normalized
        """
        B = images.shape[0]

        # Multi-scale backbone features
        multi_scale_features = self.backbone(images)  # 3 levels

        # Encode
        memory = self._encode_multi_scale(multi_scale_features)  # [B, sum(HW), D]

        # Decode
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(query, memory)  # [B, Q, D]

        # Predict
        pred_logits = self.class_head(hs)
        pred_boxes = self.bbox_head(hs).sigmoid()

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, score_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """Inference with filtering."""
        outputs = self.forward(images)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        B = pred_logits.shape[0]
        results = []

        for i in range(B):
            probs = pred_logits[i].softmax(-1)
            scores = probs[:, 1]  # Pneumonia class
            labels = torch.ones(probs.shape[0], device=probs.device, dtype=torch.long)

            keep = scores > score_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes_cxcywh = pred_boxes[i][keep]

            boxes_xyxy = box_convert(boxes_cxcywh, "cxcywh", "xyxy")
            img_h, img_w = images.shape[2], images.shape[3]
            boxes_xyxy[:, [0, 2]] *= img_w
            boxes_xyxy[:, [1, 3]] *= img_h

            results.append({
                "boxes": boxes_xyxy,
                "scores": scores,
                "labels": labels,
            })

        return results


def build_detr_multiscale(cfg: ModelConfig, pretrained: bool = True) -> MultiScaleDETR:
    """Build multi-scale DETR model."""
    return MultiScaleDETR(cfg, pretrained_backbone=pretrained)
