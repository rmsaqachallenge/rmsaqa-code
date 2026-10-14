"""AF3-style spatial replacement projector for MiDashengLM audio features."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SpatialAudioProjectorSubsample(nn.Module):
    """Fuse Dasheng semantic and spatial features before 5x temporal subsampling."""

    SUBSAMPLE_FACTOR = 5
    SEMANTIC_DIM = 1280
    SPATIAL_DIM = 1024
    HIDDEN_SIZE = 3584

    def __init__(self, official_projector: nn.Module, *, dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        try:
            official_first = official_projector.net[0]
            official_second = official_projector.net[2]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("official_projector must expose net[0] and net[2] Linear layers") from exc
        if not isinstance(official_first, nn.Linear) or not isinstance(official_second, nn.Linear):
            raise ValueError("official projector layers must be Linear modules")
        if official_first.in_features != self.SUBSAMPLE_FACTOR * self.SEMANTIC_DIM or official_first.out_features != self.HIDDEN_SIZE:
            raise ValueError("official projector first layer has incompatible dimensions")
        if official_second.in_features != self.HIDDEN_SIZE or official_second.out_features != self.HIDDEN_SIZE:
            raise ValueError("official projector second layer has incompatible dimensions")
        layer_dtype = dtype or official_first.weight.dtype
        self.net = nn.Sequential(
            nn.Linear(self.SUBSAMPLE_FACTOR * (self.SEMANTIC_DIM + self.SPATIAL_DIM), self.HIDDEN_SIZE, dtype=layer_dtype),
            nn.GELU(),
            nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE, dtype=layer_dtype),
        )
        with torch.no_grad():
            self.net[0].weight[:, : self.SUBSAMPLE_FACTOR * self.SEMANTIC_DIM].copy_(official_first.weight.to(layer_dtype))
            self.net[0].weight[:, self.SUBSAMPLE_FACTOR * self.SEMANTIC_DIM :].zero_()
            self.net[0].bias.copy_(official_first.bias.to(layer_dtype))
            self.net[2].weight.copy_(official_second.weight.to(layer_dtype))
            self.net[2].bias.copy_(official_second.bias.to(layer_dtype))
        self.k = self.SUBSAMPLE_FACTOR

    @staticmethod
    def _validate_features(value: torch.Tensor, dim: int, name: str) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != dim:
            raise ValueError(f"{name} must have shape [B, T, {dim}]")
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError(f"{name} must have non-empty batch/time dimensions")
        if not value.is_floating_point() or value.is_complex() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain finite floating-point values")

    def forward(
        self,
        semantic_features: torch.Tensor,
        spatial_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_features(semantic_features, self.SEMANTIC_DIM, "semantic_features")
        self._validate_features(spatial_features, self.SPATIAL_DIM, "spatial_features")
        if semantic_features.shape[:2] != spatial_features.shape[:2]:
            raise ValueError("semantic_features and spatial_features must share batch/time dimensions")
        b, t, _ = semantic_features.shape
        combined_mask = None
        for candidate, name in ((mask, "mask"), (spatial_mask, "spatial_mask")):
            if candidate is None:
                continue
            if not isinstance(candidate, torch.Tensor) or candidate.shape != (b, t):
                raise ValueError(f"{name} must have shape [B, T]")
            if candidate.is_complex() or not torch.all((candidate == 0) | (candidate == 1)).item():
                raise ValueError(f"{name} must be binary (0/1)")
            candidate = candidate.to(device=semantic_features.device, dtype=torch.bool)
            combined_mask = candidate if combined_mask is None else (combined_mask & candidate)
        n = t // self.k
        if n == 0:
            raise ValueError("feature sequence must contain at least 5 frames")
        semantic = semantic_features[:, : n * self.k].reshape(b, n, self.k * self.SEMANTIC_DIM)
        spatial = spatial_features[:, : n * self.k].reshape(b, n, self.k * self.SPATIAL_DIM)
        fused = torch.cat((semantic, spatial), dim=-1)
        first_layer = self.net[0]
        fused = fused.to(device=first_layer.weight.device, dtype=first_layer.weight.dtype)
        output = self.net(fused)
        if combined_mask is None:
            out_mask = torch.ones((b, n), dtype=torch.long, device=output.device)
        else:
            out_mask = combined_mask[:, : n * self.k].reshape(b, n, self.k).any(dim=-1).long()
        return output, out_mask
