"""Small, deterministic inference adapters for MiDashengLM evaluation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch

from .models.midashenglm_spatial_training import SpatialMiDashengLMTrainingModel
from .midashenglm_prompts import build_layer2_messages


def _values(batch, singular, plural):
    value = batch.get(plural, batch.get(singular))
    if value is None:
        raise ValueError(f"batch is missing {plural}")
    return value


class _Base:
    @staticmethod
    def _generate_one(
        backbone, tokens, prompt_ids, device, max_new_tokens, num_beams
    ):
        inputs = backbone.build_inputs_embeds(
            [prompt_ids], tokens, device=device
        )
        try:
            text = backbone.generate_from_inputs_embeds(
                inputs["inputs_embeds"],
                inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )
        except TypeError:
            text = backbone.generate_from_inputs_embeds(
                inputs["inputs_embeds"],
                inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        if not isinstance(text, (list, tuple)) or len(text) != 1:
            raise ValueError("generation output length mismatch")
        return text[0]

    def _generate(self, backbone, tokens, batch, device, max_new_tokens, num_beams, audio_mask=None):
        backbone.eval()
        n = tokens.shape[0]
        mask = audio_mask if audio_mask is not None else batch.get("midasheng_audio_attention_mask")
        if mask is None: mask = batch.get("spatial_attention_mask")
        counts = [int(mask[i].to(torch.bool).sum()) for i in range(n)] if mask is not None else [int(tokens.shape[1])] * n
        if len(counts) != n or any(c <= 0 or c > tokens.shape[1] for c in counts): raise ValueError("invalid audio attention mask")
        q1 = _values(batch, "layer1_questions", "layer1_questions")
        q2 = _values(batch, "layer2_questions", "layer2_questions")
        o1 = batch.get("layer1_options", [None] * n)
        o2 = batch.get("layer2_options", [None] * n)
        if any(len(values) != n for values in (q1, q2, o1, o2)):
            raise ValueError("question/options batch length mismatch")

        layer1_text = []
        for i in range(n):
            prompt = backbone.build_user_prompt(q1[i], o1[i])
            prompt_ids = backbone.build_chat_prompt_ids(prompt, counts[i])
            layer1_text.append(
                self._generate_one(
                    backbone,
                    tokens[i:i+1, :counts[i]],
                    prompt_ids,
                    device,
                    max_new_tokens,
                    num_beams,
                )
            )

        layer2_text = []
        for i in range(n):
            messages = build_layer2_messages(
                backbone,
                layer1_question=q1[i],
                layer1_options=o1[i],
                layer1_answer=layer1_text[i],
                layer2_question=q2[i],
                layer2_options=o2[i],
            )
            prompt_ids = backbone.build_chat_prompt_ids_from_messages(
                messages, counts[i]
            )
            layer2_text.append(
                self._generate_one(
                    backbone,
                    tokens[i:i+1, :counts[i]],
                    prompt_ids,
                    device,
                    max_new_tokens,
                    num_beams,
                )
            )
        return {
            "layer1_text": layer1_text,
            "layer2_text": layer2_text,
            "audio_token_count": counts,
        }


class SpatialMiDashengLMEvaluator(_Base):
    def __init__(self, midashenglm_root: str, *, checkpoint: str | Path, online_waveform=False, **kwargs):
        self.online_waveform = bool(online_waveform)
        self.audio_encoder_device = kwargs.pop("audio_encoder_device", None)
        self.model = SpatialMiDashengLMTrainingModel(midashenglm_root, **kwargs)
        if self.audio_encoder_device:
            encoder = getattr(getattr(self.model, "backbone", None), "audio_encoder", None)
            if encoder is not None and hasattr(encoder, "to"): encoder.to(self.audio_encoder_device)
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        expected = payload.get("trainable_parameter_names") if isinstance(payload, dict) else None
        names_fn = getattr(self.model, "trainable_parameter_names", None)
        actual = set(names_fn() if callable(names_fn) else (n for n, p in self.model.named_parameters() if p.requires_grad))
        if expected is not None and set(expected) != actual:
            raise ValueError("trainable_parameter_names mismatch")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        known = set(self.model.state_dict())
        unknown = set(state) - known
        if any(name in actual for name in missing) or unexpected or unknown:
            raise ValueError("trainable checkpoint does not match model parameters")

    def generate_freeform_answers(self, batch, *, device="cpu", max_new_tokens=12, num_beams=1):
        self.model.eval()
        if hasattr(self.model, "to"): self.model.to(device)
        spatial = batch["spatial_features"].to(device).float()
        if hasattr(self.model, "spatial_encoder"):
            spatial = self.model.spatial_encoder(spatial)
        if self.online_waveform:
            wave = batch.get("input_values")
            if wave is None or wave.ndim != 2: raise ValueError("online waveform requires preprocessed W-channel input_values")
            bundle = self.model.backbone.encode_spatial_audio_bundle(wave.to(device), spatial, batch.get("audio_length"))
            tokens, token_mask = bundle["spatial_tokens"], bundle.get("spatial_mask")
        else:
            bundle = self.model.backbone.encode_cached_spatial_audio_bundle(
                batch["dasheng_features"].to(device), batch["dasheng_attention_mask"].to(device), spatial)
            tokens, token_mask = bundle["spatial_tokens"], bundle.get("spatial_mask")
        return self._generate(self.model.backbone, tokens, batch, device, max_new_tokens, num_beams, token_mask)
