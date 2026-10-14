from __future__ import annotations

from typing import Any, Optional, Sequence
import torch
import torch.nn.functional as F
from torch import nn

from .midashenglm_generative_backbone import FrozenMiDashengLMGenerativeBackbone
from .midashenglm_spatial_projector import SpatialAudioProjectorSubsample


class SpatialMiDashengLMGenerativeBackbone(nn.Module):
    def __init__(self, midashenglm_root: str, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = FrozenMiDashengLMGenerativeBackbone(midashenglm_root, **kwargs)
        self.spatial_projector = SpatialAudioProjectorSubsample(self.backbone.audio_projector)
        for p in self.spatial_projector.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode)
        self.spatial_projector.train(mode)
        self.backbone.audio_encoder.eval()
        self.backbone.audio_projector.eval()
        return self

    @property
    def tokenizer(self):
        return self.backbone.tokenizer

    @property
    def eos_token_id(self):
        return self.backbone.eos_token_id

    def build_user_prompt(self, *args, **kwargs): return self.backbone.build_user_prompt(*args, **kwargs)
    def build_chat_prompt_ids(self, *args, **kwargs): return self.backbone.build_chat_prompt_ids(*args, **kwargs)
    def build_chat_prompt_ids_from_messages(self, *args, **kwargs): return self.backbone.build_chat_prompt_ids_from_messages(*args, **kwargs)
    def build_inputs_embeds(self, *args, **kwargs): return self.backbone.build_inputs_embeds(*args, **kwargs)
    def forward_lm(self, *args, **kwargs): return self.backbone.forward_lm(*args, **kwargs)
    def generate_from_inputs_embeds(self, *args, **kwargs): return self.backbone.generate_from_inputs_embeds(*args, **kwargs)

    def trainable_parameter_names(self) -> list[str]:
        names = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                names.append(n)
        return names

    def encode_spatial_audio_bundle(self, input_values: torch.Tensor, spatial_features: torch.Tensor, audio_length: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor | None]:
        if not isinstance(spatial_features, torch.Tensor) or spatial_features.ndim != 3 or spatial_features.shape[-1] != 1024:
            raise ValueError("spatial_features must have shape [B, T, 1024]")
        if spatial_features.shape[0] == 0 or spatial_features.shape[1] == 0 or not spatial_features.is_floating_point() or not torch.isfinite(spatial_features).all().item():
            raise ValueError("spatial_features must be non-empty finite floating-point values")
        bundle = self.backbone._encode_audio_bundle(input_values, audio_length, include_native=False)
        raw = bundle["raw_features"]
        if raw.shape[0] != spatial_features.shape[0]:
            raise ValueError("spatial_features batch must match audio batch")
        spatial_features = spatial_features.to(device=raw.device)
        aligned = F.interpolate(spatial_features.transpose(1, 2), size=raw.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        tokens, mask = self.spatial_projector(raw, aligned, bundle.get("raw_mask"))
        return {"raw_features": raw, "raw_mask": bundle.get("raw_mask"), "aligned_spatial_features": aligned, "spatial_tokens": tokens, "spatial_mask": mask}

    def encode_spatial_audio_tokens(self, input_values: torch.Tensor, spatial_features: torch.Tensor, audio_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode_spatial_audio_bundle(input_values, spatial_features, audio_length)["spatial_tokens"]  # type: ignore[return-value]

    def encode_cached_spatial_audio_bundle(self, dasheng_features: torch.Tensor, dasheng_attention_mask: torch.Tensor, spatial_features: torch.Tensor):
        if not isinstance(dasheng_features, torch.Tensor) or dasheng_features.ndim != 3 or dasheng_features.shape[-1] != 1280 or dasheng_features.shape[0] == 0 or dasheng_features.shape[1] == 0 or not dasheng_features.is_floating_point() or not torch.isfinite(dasheng_features).all().item(): raise ValueError("dasheng_features must be finite [B,T,1280]")
        if not isinstance(spatial_features, torch.Tensor) or spatial_features.ndim != 3 or spatial_features.shape[-1] != 1024 or spatial_features.shape[0] != dasheng_features.shape[0] or spatial_features.shape[1] == 0 or not spatial_features.is_floating_point() or not torch.isfinite(spatial_features).all().item(): raise ValueError("spatial_features shape/batch mismatch")
        if not isinstance(dasheng_attention_mask, torch.Tensor): raise ValueError("dasheng_attention_mask must be binary")
        integer_mask = dasheng_attention_mask.dtype == torch.bool or not dasheng_attention_mask.dtype.is_floating_point
        if not integer_mask or dasheng_attention_mask.shape != dasheng_features.shape[:2] or dasheng_attention_mask.is_complex() or not torch.all((dasheng_attention_mask == 0) | (dasheng_attention_mask == 1)).item(): raise ValueError("dasheng_attention_mask must be binary")
        spatial_features = spatial_features.to(dasheng_features.device)
        dasheng_attention_mask = dasheng_attention_mask.to(device=dasheng_features.device, dtype=torch.long)
        aligned = F.interpolate(spatial_features.transpose(1,2), size=dasheng_features.shape[1], mode="linear", align_corners=False).transpose(1,2)
        tokens, mask = self.spatial_projector(dasheng_features, aligned, dasheng_attention_mask)
        return {"raw_features": dasheng_features, "raw_mask": dasheng_attention_mask, "aligned_spatial_features": aligned, "spatial_tokens": tokens, "spatial_mask": mask}

    def encode_cached_spatial_audio_tokens(self, dasheng_features, dasheng_attention_mask, spatial_features):
        return self.encode_cached_spatial_audio_bundle(dasheng_features, dasheng_attention_mask, spatial_features)["spatial_tokens"]

    def build_spatial_inputs_embeds(self, prompt_ids_list: Sequence[torch.Tensor], input_values: torch.Tensor, spatial_features: torch.Tensor, answer_ids_list: Optional[Sequence[Optional[torch.Tensor]]] = None, device: torch.device | str = "cpu", audio_length: Optional[torch.Tensor] = None):
        tokens = self.encode_spatial_audio_tokens(input_values, spatial_features, audio_length)
        return self.build_inputs_embeds(prompt_ids_list, tokens, answer_ids_list=answer_ids_list, device=device)
