"""
Domain-Adaptive Pretraining via Masked Autoencoder (MAE).

Continues self-supervised pretraining on unlabeled chest X-rays
to adapt ImageNet features to the radiographic domain.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class MAEEncoder(nn.Module):
    """
    Simplified MAE encoder based on ResNet-50.
    Patches are created via conv layers, then randomly masked.
    """

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.proj = nn.Conv2d(2048, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> [B, D, H', W']"""
        return self.proj(self.features(x))


class MAEDecoder(nn.Module):
    """Lightweight decoder to reconstruct masked patches."""

    def __init__(
        self,
        encoder_dim: int = 256,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        patch_size: int = 32,  # Effective patch size from ResNet
    ):
        super().__init__()
        self.embed_proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Predict pixel values for each patch
        self.pred_head = nn.Linear(decoder_dim, patch_size * patch_size * 3)
        self.patch_size = patch_size

    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoded: [B, N_visible, encoder_dim] visible patch embeddings
            mask: [B, N_total] boolean mask (True = masked)

        Returns:
            pred_pixels: [B, N_total, patch_size^2 * 3] reconstructed patches
        """
        B, N_vis, D = encoded.shape
        N_total = mask.shape[1]
        N_mask = N_total - N_vis

        # Project encoder output
        visible_tokens = self.embed_proj(encoded)  # [B, N_vis, decoder_dim]

        # Create full sequence with mask tokens
        mask_tokens = self.mask_token.expand(B, N_mask, -1)  # [B, N_mask, decoder_dim]

        # Merge: put visible and mask tokens in correct positions
        full_seq = torch.zeros(
            B, N_total, visible_tokens.shape[-1],
            device=encoded.device, dtype=encoded.dtype
        )
        visible_idx = (~mask).nonzero(as_tuple=True)
        masked_idx = mask.nonzero(as_tuple=True)

        full_seq[visible_idx[0], visible_idx[1]] = visible_tokens.reshape(-1, visible_tokens.shape[-1])
        full_seq[masked_idx[0], masked_idx[1]] = mask_tokens.reshape(-1, mask_tokens.shape[-1])

        # Decode
        decoded = self.decoder(full_seq)  # [B, N_total, decoder_dim]

        # Predict pixels
        pred_pixels = self.pred_head(decoded)  # [B, N_total, patch_size^2 * 3]

        return pred_pixels


class MaskedAutoencoder(nn.Module):
    """
    MAE for domain-adaptive pretraining on chest X-rays.

    Training procedure:
    1. Patchify image into non-overlapping patches
    2. Randomly mask 75% of patches
    3. Encode visible patches with ResNet backbone
    4. Decode and reconstruct masked patches
    5. Compute MSE loss on masked patches only
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        mask_ratio: float = 0.75,
        image_size: int = 224,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = 32  # ResNet effective stride

        self.encoder = MAEEncoder(hidden_dim, pretrained_backbone)
        self.decoder = MAEDecoder(
            hidden_dim, decoder_dim, decoder_depth, decoder_heads, self.patch_size
        )

        self.n_patches_h = image_size // self.patch_size
        self.n_patches_w = image_size // self.patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

    def random_masking(self, B: int, device: torch.device):
        """Generate random mask. True = masked (to reconstruct)."""
        n_mask = int(self.n_patches * self.mask_ratio)
        noise = torch.rand(B, self.n_patches, device=device)
        ids_shuffle = noise.argsort(dim=1)
        mask = torch.zeros(B, self.n_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)
        return mask

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patches: [B, 3, H, W] -> [B, N, P*P*3]"""
        B, C, H, W = images.shape
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)  # [B, 3, n_h, n_w, p, p]
        patches = patches.contiguous().view(B, C, -1, p, p)  # [B, 3, N, p, p]
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, -1, C * p * p)  # [B, N, P*P*3]
        return patches

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            loss: MSE reconstruction loss on masked patches
            pred: predicted patches
            mask: which patches were masked
        """
        B = images.shape[0]
        device = images.device

        # Generate mask
        mask = self.random_masking(B, device)  # [B, N]

        # Encode (full image through backbone, then select visible)
        features = self.encoder(images)  # [B, D, H', W']
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, N, D]

        # Select visible tokens
        visible_mask = ~mask
        visible_tokens = []
        for i in range(B):
            visible_tokens.append(features_flat[i, visible_mask[i]])
        visible_tokens = torch.stack(visible_tokens)  # [B, N_vis, D]

        # Decode
        pred_pixels = self.decoder(visible_tokens, mask)  # [B, N, P*P*3]

        # Compute loss on masked patches only
        target_patches = self.patchify(images)  # [B, N, P*P*3]

        # Normalize target per patch
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        target_norm = (target_patches - mean) / (var + 1e-6).sqrt()

        loss = F.mse_loss(pred_pixels[mask], target_norm[mask])

        return loss, pred_pixels, mask

    def get_encoder_state_dict(self):
        """Extract encoder weights for downstream detection task."""
        return self.encoder.state_dict()


def pretrain_mae(
    model: MaskedAutoencoder,
    dataloader,
    optimizer,
    scheduler,
    num_epochs: int,
    device: str = "cuda",
    save_path: str = "./outputs/mae_pretrained.pth",
):
    """Run MAE pretraining loop."""
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        n_batches = 0

        for batch_idx, images in enumerate(dataloader):
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)

            loss, _, _ = model(images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, "
                      f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

    # Save pretrained encoder
    torch.save(model.get_encoder_state_dict(), save_path)
    print(f"Pretrained encoder saved to {save_path}")

    return model
