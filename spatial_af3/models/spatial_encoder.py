from __future__ import annotations

import torch
import torch.nn as nn

from spatial_models.conformer import ConformerBlock
from spatial_models.resnet import resnet18_nopool


class ResnetConformerTokenEncoder(nn.Module):
    def __init__(self, in_channels: int = 7, mel_bins: int = 64, encoder_dim: int = 1024, num_layers: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.mel_bins = mel_bins
        self.resnet = resnet18_nopool(in_channel=in_channels)
        embedding_dim = mel_bins // 32 * 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        self.conformer_layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=encoder_dim,
                    dim_head=32,
                    heads=8,
                    ff_mult=2,
                    conv_expansion_factor=2,
                    conv_kernel_size=7,
                    attn_dropout=0.1,
                    ff_dropout=0.1,
                    conv_dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 7*64]
        bsz, time_steps, feat_dim = x.shape
        assert feat_dim == self.in_channels * self.mel_bins, f"Expected feat_dim={self.in_channels * self.mel_bins}, got {feat_dim}"
        x = x.view(bsz, time_steps, self.in_channels, self.mel_bins).permute(0, 2, 1, 3)
        conv_outputs = self.resnet(x)
        n, c, t, f = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(n, t, c * f)
        hidden = self.input_projection(conv_outputs)
        for layer in self.conformer_layers:
            hidden = layer(hidden)
        return self.out_norm(hidden)
