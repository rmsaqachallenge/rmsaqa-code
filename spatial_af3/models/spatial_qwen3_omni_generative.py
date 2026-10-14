from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import operator
import re
import threading
import weakref
from numbers import Real
from typing import Dict, List, Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_omni_generative_backbone import FrozenQwen3OmniGenerativeBackbone
from .spatial_encoder import ResnetConformerTokenEncoder
from ..layered_mcq import build_mcq_prompt_body, format_mcq_target


_TRAINABLE_BACKBONE_LORA_PARAMETER = re.compile(
    r"^backbone\..*\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)\.lora_[AB]\.[^.]+\.weight$"
)

_WRAPPER_STATE_LOCKS_GUARD = threading.Lock()
_WRAPPER_STATE_LOCKS: weakref.WeakKeyDictionary[
    nn.Module, threading.RLock
] = weakref.WeakKeyDictionary()


class Qwen3OmniFreeformGenerationOutput(TypedDict):
    layer1_text: List[str]
    layer2_text: List[str]
    layer1_gate: List[float]
    layer2_gate: List[float]


class Qwen3OmniTemporalConcatFusion(nn.Module):
    def __init__(
        self,
        semantic_dim: int = 1280,
        spatial_dim: int = 1024,
        llm_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.semantic_dim = self._validate_dimension(semantic_dim, "semantic_dim")
        self.spatial_dim = self._validate_dimension(spatial_dim, "spatial_dim")
        self.llm_dim = self._validate_dimension(llm_dim, "llm_dim")
        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        self.concat_project = nn.Linear(self.semantic_dim + self.spatial_dim, self.llm_dim)
        self._init_concat_project()

    def _init_concat_project(self) -> None:
        with torch.no_grad():
            self.concat_project.weight.zero_()
            self.concat_project.bias.zero_()
            nn.init.kaiming_uniform_(
                self.concat_project.weight[:, : self.semantic_dim],
                a=math.sqrt(5),
            )

    @staticmethod
    def _validate_dimension(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a positive integer")
        try:
            dimension = operator.index(value)
        except TypeError as error:
            raise TypeError(f"{name} must be a positive integer") from error
        if dimension <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return dimension

    def forward(
        self,
        semantic_audio: torch.Tensor,
        spatial_seq: torch.Tensor,
        semantic_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs(semantic_audio, spatial_seq)
        audio_content_mask = self._validate_mask(
            semantic_attention_mask,
            batch_size=semantic_audio.shape[0],
            semantic_time=semantic_audio.shape[1],
            device=semantic_audio.device,
        )
        return self._forward_validated(semantic_audio, spatial_seq, audio_content_mask)

    def _forward_validated(
        self,
        semantic_audio: torch.Tensor,
        spatial_seq: torch.Tensor,
        audio_content_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        aligned_spatial = F.adaptive_avg_pool1d(
            spatial_seq.transpose(1, 2), output_size=semantic_audio.shape[1]
        ).transpose(1, 2)
        normalized_spatial = self.spatial_norm(aligned_spatial)
        fused_audio = self.concat_project(
            torch.cat((semantic_audio, normalized_spatial), dim=-1)
        )
        return {
            "fused_audio": fused_audio,
            "aligned_spatial": aligned_spatial,
            "normalized_spatial": normalized_spatial,
            "audio_content_mask": audio_content_mask,
            "gate": torch.ones(
                semantic_audio.shape[:2],
                dtype=semantic_audio.dtype,
                device=semantic_audio.device,
            ),
        }

    def _validate_inputs(
        self, semantic_audio: torch.Tensor, spatial_seq: torch.Tensor
    ) -> None:
        for name, value in (("semantic_audio", semantic_audio), ("spatial_seq", spatial_seq)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.layout != torch.strided:
                raise ValueError(f"{name} must use strided layout")
            if value.device.type == "meta":
                raise ValueError(f"{name} must not use the meta device")
            if value.is_quantized:
                raise ValueError(f"{name} must not be quantized")
            if value.ndim != 3:
                raise ValueError(f"{name} must have shape [batch, time, hidden]")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")

        if semantic_audio.shape[0] == 0 or spatial_seq.shape[0] == 0:
            raise ValueError("batch dimensions must be non-empty")
        if semantic_audio.shape[1] == 0 or spatial_seq.shape[1] == 0:
            raise ValueError("time dimensions must be non-empty")
        if semantic_audio.shape[0] != spatial_seq.shape[0]:
            raise ValueError("semantic_audio and spatial_seq batch sizes must match")
        if semantic_audio.shape[2] != self.semantic_dim:
            raise ValueError(f"expected semantic_dim={self.semantic_dim}")
        if spatial_seq.shape[2] != self.spatial_dim:
            raise ValueError(f"expected spatial_dim={self.spatial_dim}")
        if semantic_audio.device != spatial_seq.device:
            raise ValueError("semantic_audio and spatial_seq must share a device")
        if semantic_audio.dtype != spatial_seq.dtype:
            raise ValueError("semantic_audio and spatial_seq must share a dtype")

    @staticmethod
    def _validate_mask(
        mask: Optional[torch.Tensor],
        batch_size: int,
        semantic_time: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones((batch_size, semantic_time), dtype=torch.bool, device=device)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("semantic_attention_mask must be a torch.Tensor")
        if tuple(mask.shape) != (batch_size, semantic_time):
            raise ValueError("semantic_attention_mask must have shape [batch, semantic_time]")
        if mask.layout != torch.strided:
            raise TypeError("semantic_attention_mask must use strided layout")
        if mask.device.type == "meta":
            raise ValueError("semantic_attention_mask must not use the meta device")
        if mask.is_quantized:
            raise TypeError("semantic_attention_mask must not be quantized")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("semantic_attention_mask must be binary")
        mask = mask.to(device=device, dtype=torch.bool)
        if semantic_time > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("semantic_attention_mask must be left-aligned")
        return mask


class SpatialQwen3OmniGenerativeModel(nn.Module):
    def __init__(
        self,
        qwen3_omni_root: str,
        text_dtype: torch.dtype = torch.bfloat16,
        audio_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        enable_llm_lora: bool = True,
        llm_lora_r: int = 12,
        llm_lora_alpha: int = 12,
        llm_lora_dropout: float = 0.05,
        llm_lora_num_layers: int = 48,
        spatial_encoder_dim: int = 1024,
        loss_weights: Optional[Mapping[str, Real]] = None,
    ) -> None:
        super().__init__()
        spatial_encoder_dim = self._validate_positive_integer(
            spatial_encoder_dim, "spatial_encoder_dim"
        )
        self.loss_weights = self._validate_loss_weights(loss_weights)
        self.backbone = FrozenQwen3OmniGenerativeBackbone(
            qwen3_omni_root=qwen3_omni_root,
            text_dtype=text_dtype,
            audio_dtype=audio_dtype,
            attn_implementation=attn_implementation,
            enable_llm_lora=enable_llm_lora,
            llm_lora_r=llm_lora_r,
            llm_lora_alpha=llm_lora_alpha,
            llm_lora_dropout=llm_lora_dropout,
            llm_lora_num_layers=llm_lora_num_layers,
        )
        self._trusted_backbone_trainable_parameters = (
            self._capture_trusted_backbone_trainable_parameters(enable_llm_lora)
        )
        self.spatial_encoder = ResnetConformerTokenEncoder(
            encoder_dim=spatial_encoder_dim,
        )
        spatial_in_channels = self._validate_positive_integer(
            getattr(self.spatial_encoder, "in_channels", None),
            "spatial_encoder.in_channels",
        )
        spatial_mel_bins = self._validate_positive_integer(
            getattr(self.spatial_encoder, "mel_bins", None),
            "spatial_encoder.mel_bins",
        )
        self.spatial_input_dim = spatial_in_channels * spatial_mel_bins
        self.fusion = Qwen3OmniTemporalConcatFusion(
            semantic_dim=self.backbone.audio_hidden_size,
            spatial_dim=spatial_encoder_dim,
            llm_dim=self.backbone.hidden_size,
        )
        self._audit_trainable_parameters()
        self.train(self.training)

    @staticmethod
    def _validate_loss_weights(
        loss_weights: Optional[Mapping[str, Real]],
    ) -> Dict[str, float]:
        validated = {"lm_layer1": 1.0, "lm_layer2": 1.0}
        if loss_weights is not None:
            if not isinstance(loss_weights, Mapping):
                raise TypeError("loss_weights must be a mapping")
            for name in loss_weights:
                if not isinstance(name, str):
                    raise TypeError(f"loss_weights key must be a string: {name!r}")
            unknown = sorted(set(loss_weights) - set(validated))
            if unknown:
                raise ValueError("unknown loss weight key(s): " + ", ".join(unknown))
            for name, weight in loss_weights.items():
                if isinstance(weight, bool):
                    raise ValueError(f"loss weight {name!r} must not be bool")
                if not isinstance(weight, Real):
                    raise ValueError(f"loss weight {name!r} must be a real number")
                if not math.isfinite(weight):
                    raise ValueError(f"loss weight {name!r} must be finite")
                if weight < 0:
                    raise ValueError(f"loss weight {name!r} must be non-negative")
                validated[name] = float(weight)
        if not any(weight > 0 for weight in validated.values()):
            raise ValueError("at least one loss weight must be greater than zero")
        return validated

    def _capture_trusted_backbone_trainable_parameters(
        self, enable_llm_lora: bool
    ) -> dict[str, int]:
        trainable = {
            f"backbone.{name}": parameter
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        if not enable_llm_lora and trainable:
            raise RuntimeError(
                "backbone LoRA is disabled but trainable parameter(s) remain: "
                + ", ".join(sorted(trainable))
            )
        invalid = sorted(
            name
            for name in trainable
            if _TRAINABLE_BACKBONE_LORA_PARAMETER.fullmatch(name) is None
        )
        if invalid:
            raise RuntimeError(
                "unexpected trainable backbone parameter(s): " + ", ".join(invalid)
            )
        return {name: id(parameter) for name, parameter in trainable.items()}

    def _audit_trainable_parameters(self) -> None:
        current_backbone = {
            f"backbone.{name}": id(parameter)
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        trusted = self._trusted_backbone_trainable_parameters
        extra = sorted(set(current_backbone) - set(trusted))
        missing = sorted(set(trusted) - set(current_backbone))
        replaced = sorted(
            name
            for name in set(current_backbone) & set(trusted)
            if current_backbone[name] != trusted[name]
        )
        unexpected = sorted(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and not (
                name.startswith("backbone.")
                or name.startswith("spatial_encoder.")
                or name.startswith("fusion.")
            )
        )
        problems = []
        for label, names in (
            ("extra", extra),
            ("missing", missing),
            ("replaced", replaced),
            ("unexpected", unexpected),
        ):
            if names:
                problems.append(f"{label}: {', '.join(names)}")
        if problems:
            raise RuntimeError("trainable parameter audit failed; " + "; ".join(problems))

    def train(self, mode: bool = True) -> "SpatialQwen3OmniGenerativeModel":
        with self._state_lock():
            super().train(mode)
            self._audit_trainable_parameters()
        return self

    def _state_lock(self) -> threading.RLock:
        with _WRAPPER_STATE_LOCKS_GUARD:
            lock = _WRAPPER_STATE_LOCKS.get(self)
            if lock is None:
                lock = threading.RLock()
                _WRAPPER_STATE_LOCKS[self] = lock
            return lock

    @staticmethod
    def _normalize_history_text(text: object) -> str:
        return " ".join(str(text or "").strip().split())

    @staticmethod
    def _assistant_message(text: str) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        }

    @staticmethod
    def _normalize_mcq_inputs(
        question: object, options: object
    ) -> tuple[str, dict[str, str]]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must be non-empty after stripping")
        if not isinstance(options, Mapping) or not options:
            raise TypeError("options must be a non-empty mapping")
        normalized_options: dict[str, str] = {}
        for raw_key, raw_text in options.items():
            key = str(raw_key).strip().upper()
            if not key:
                raise ValueError("option key must be non-empty after stripping")
            if key in normalized_options:
                raise ValueError(f"duplicate option key after normalization: {key}")
            text = str(raw_text).strip()
            if not text:
                raise ValueError(f"option text for {key} must be non-empty after stripping")
            normalized_options[key] = text
        return normalized_question, normalized_options

    def _normalize_training_mcq_sample(
        self, question: object, options: object, correct_option: object
    ) -> tuple[str, dict[str, str], str]:
        normalized_question, normalized_options = self._normalize_mcq_inputs(
            question, options
        )
        normalized_correct_option = str(correct_option or "").strip().upper()
        if not normalized_correct_option:
            raise ValueError("MCQ correct_option is required")
        if normalized_correct_option not in normalized_options:
            raise ValueError(
                f"Correct option {normalized_correct_option!r} is not present in options: "
                f"{sorted(normalized_options)}"
            )
        return normalized_question, normalized_options, normalized_correct_option

    @classmethod
    def _followup_user_message(cls, question: object, options: object) -> dict[str, object]:
        question, options = cls._normalize_mcq_inputs(question, options)
        return cls._followup_user_message_from_normalized(question, options)

    @staticmethod
    def _followup_user_message_from_normalized(
        question: str, options: Mapping[str, str]
    ) -> dict[str, object]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Based on the same audio clip and the previous answer, "
                        "answer the follow-up question.\n"
                        + build_mcq_prompt_body(question, options)
                    ),
                }
            ],
        }

    def _build_layer1_messages(
        self, question: object, options: object
    ) -> list[object]:
        question, options = self._normalize_mcq_inputs(question, options)
        return self._build_layer1_messages_from_normalized(question, options)

    def _build_layer1_messages_from_normalized(
        self, question: str, options: Mapping[str, str]
    ) -> list[object]:
        messages = self.backbone.build_user_messages(question, options)
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise TypeError("build_user_messages must return a non-string sequence")
        if not messages:
            raise ValueError("build_user_messages must return a non-empty sequence")
        copied_messages = deepcopy(list(messages))
        audio_count = 0
        for message in copied_messages:
            if not isinstance(message, Mapping):
                raise TypeError("messages entries must be mappings")
            role = message.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError("messages entries must have a non-empty string role")
            content = message.get("content")
            if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
                raise TypeError("messages content must be a non-string sequence")
            if not content:
                raise ValueError("messages content must be non-empty")
            for item in content:
                if not isinstance(item, Mapping):
                    raise TypeError("messages content items must be mappings")
                if "type" not in item:
                    raise ValueError("messages content items must include type")
                content_type = item["type"]
                if not isinstance(content_type, str):
                    raise TypeError("messages content item type must be a string")
                if content_type not in {"audio", "text"}:
                    raise ValueError("messages content item type must be audio or text")
                field_name = "audio" if content_type == "audio" else "text"
                field_value = item.get(field_name)
                if not isinstance(field_value, str):
                    raise TypeError(
                        f"messages {content_type} items must contain a string {field_name}"
                    )
                if not field_value.strip():
                    raise ValueError(
                        f"messages {content_type} items must contain a non-empty {field_name}"
                    )
                audio_count += content_type == "audio"
        if audio_count != 1:
            raise ValueError("layer1 messages must contain exactly one audio placeholder")
        return copied_messages

    def _extend_layer1_messages(
        self, layer1_messages: Sequence[object], layer1_answer: object,
        layer2_question: object, layer2_options: object,
    ) -> list[object]:
        layer2_question, layer2_options = self._normalize_mcq_inputs(
            layer2_question, layer2_options
        )
        return self._extend_layer1_messages_from_normalized(
            layer1_messages, layer1_answer, layer2_question, layer2_options
        )

    def _extend_layer1_messages_from_normalized(
        self, layer1_messages: Sequence[object], layer1_answer: object,
        layer2_question: str, layer2_options: Mapping[str, str],
    ) -> list[object]:
        if isinstance(layer1_messages, (str, bytes)) or not isinstance(layer1_messages, Sequence):
            raise TypeError("layer1_messages must be a non-string sequence")
        messages = deepcopy(list(layer1_messages))
        messages.append(self._assistant_message(self._normalize_history_text(layer1_answer)))
        messages.append(
            self._followup_user_message_from_normalized(
                layer2_question, layer2_options
            )
        )
        return messages

    @staticmethod
    def _validated_rendered_ids(value: object, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.layout != torch.strided:
            raise TypeError(f"{name} must use strided layout")
        if value.device.type == "meta":
            raise ValueError(f"{name} must not use the meta device")
        if value.is_quantized:
            raise TypeError(f"{name} must not be quantized")
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.numel() == 0:
            raise ValueError(f"{name} must be non-empty")
        if value.dtype == torch.bool:
            raise TypeError(f"{name} must have an integer dtype")
        try:
            torch.iinfo(value.dtype)
        except TypeError as error:
            raise TypeError(f"{name} must have an integer dtype") from error
        return value.detach().to(device="cpu", dtype=torch.long).contiguous()

    @staticmethod
    def _validate_positive_integer(value: object, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            normalized = operator.index(value)
        except TypeError as error:
            raise TypeError(f"{name} must be a positive integer") from error
        if normalized <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return normalized

    def _render_prompt_ids(
        self, messages: Sequence[object], audio_output_length: object
    ) -> torch.Tensor:
        audio_output_length = self._validate_positive_integer(
            audio_output_length, "audio_output_length"
        )
        ids = self.backbone.build_chat_prompt_ids_from_messages(
            messages,
            audio_output_length=audio_output_length,
            add_generation_prompt=True,
        )
        return self._validated_rendered_ids(ids, "prompt_ids")

    def _render_training_ids(
        self,
        messages: Sequence[object],
        answer: object,
        audio_output_length: object,
        sample_index: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_output_length = self._validate_positive_integer(
            audio_output_length, "audio_output_length"
        )
        if not isinstance(answer, str):
            raise TypeError("answer must be a string")
        if not answer.strip():
            raise ValueError("answer must be non-empty after stripping")
        if isinstance(sample_index, bool):
            raise ValueError("sample_index must be a non-negative integer")
        try:
            sample_index = operator.index(sample_index)
        except TypeError as error:
            raise TypeError("sample_index must be a non-negative integer") from error
        if sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        try:
            prompt_ids = self._render_prompt_ids(messages, audio_output_length)
        except (TypeError, ValueError) as error:
            raise type(error)(f"sample index {sample_index}: {error}") from error
        full_messages = list(messages)
        full_messages.append(self._assistant_message(answer))
        try:
            full_ids = self.backbone.build_chat_prompt_ids_from_messages(
                full_messages,
                audio_output_length=audio_output_length,
                add_generation_prompt=False,
            )
            full_ids = self._validated_rendered_ids(full_ids, "full_ids")
        except (TypeError, ValueError) as error:
            raise type(error)(f"sample index {sample_index}: {error}") from error
        prompt_length = prompt_ids.numel()
        full_length = full_ids.numel()
        if full_length <= prompt_length or not torch.equal(full_ids[:prompt_length], prompt_ids):
            raise ValueError(
                f"sample index {sample_index}: prompt/full token mismatch "
                f"(prompt length={prompt_length}, full length={full_length})"
            )
        return prompt_ids, full_ids[prompt_length:]

    def _encode_fused_audio(
        self, batch: Mapping[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        canonical_device = self._normalize_target_device(device)
        return self._encode_fused_audio_on_canonical_device(batch, canonical_device)

    def _encode_fused_audio_on_canonical_device(
        self, batch: Mapping[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        mel_features = batch.get("whisper_input_features")
        spatial_features = batch.get("spatial_features")
        self._validate_floating_sequence(
            mel_features, "whisper_input_features", "[batch, mel, time]"
        )
        self._validate_floating_sequence(
            spatial_features, "spatial_features", "[batch, time, hidden]"
        )
        if mel_features.shape[0] != spatial_features.shape[0]:
            raise ValueError("whisper_input_features and spatial_features batch sizes must match")
        actual_spatial_input_dim = spatial_features.shape[2]
        if actual_spatial_input_dim != self.spatial_input_dim:
            raise ValueError(
                "spatial_features hidden width mismatch: "
                f"expected {self.spatial_input_dim}, actual {actual_spatial_input_dim}"
            )
        spatial_parameter = next(self.spatial_encoder.parameters(), None)
        if spatial_parameter is not None and spatial_parameter.device != device:
            raise ValueError(
                f"spatial_encoder parameter device {spatial_parameter.device} "
                f"must match target device {device}"
            )

        mel_mask = batch.get("whisper_attention_mask")
        if mel_mask is None:
            mel_mask = torch.ones(
                (mel_features.shape[0], mel_features.shape[2]),
                dtype=torch.long,
                device=mel_features.device,
            )
        else:
            self._validate_whisper_attention_mask(
                mel_mask,
                batch_size=mel_features.shape[0],
                time=mel_features.shape[2],
                source_device=mel_features.device,
            )

        mel_features = mel_features.to(device)
        mel_mask = mel_mask.to(device=device, dtype=torch.long)
        output_lengths = self.backbone.compute_audio_output_lengths(mel_mask)
        if (
            not isinstance(output_lengths, torch.Tensor)
            or output_lengths.ndim != 1
            or output_lengths.shape[0] != mel_features.shape[0]
        ):
            raise ValueError("output_lengths must be an integer tensor with shape [batch]")
        if output_lengths.dtype == torch.bool:
            raise ValueError("output_lengths must be an integer tensor with shape [batch]")
        try:
            torch.iinfo(output_lengths.dtype)
        except TypeError as error:
            raise ValueError(
                "output_lengths must be an integer tensor with shape [batch]"
            ) from error
        output_lengths = output_lengths.to(device=device, dtype=torch.long)
        if torch.any(output_lengths <= 0):
            raise ValueError("output_lengths must be positive")

        with torch.no_grad():
            semantic_audio = self.backbone.encode_audio_features(
                mel_features, feature_attention_mask=mel_mask, device=device
            )
        self._validate_floating_sequence(
            semantic_audio, "semantic_audio", "[batch, time, hidden]"
        )
        semantic_audio = semantic_audio.to(dtype=torch.float32)
        max_time = int(output_lengths.max().item())
        if semantic_audio.shape[0] != mel_features.shape[0]:
            raise ValueError("semantic_audio batch size must match whisper_input_features")
        if semantic_audio.shape[1] != max_time:
            raise ValueError("semantic_audio time dimension must match output_lengths")
        if semantic_audio.shape[2] != self.backbone.audio_hidden_size:
            raise ValueError("semantic_audio hidden dimension must match backbone.audio_hidden_size")
        if semantic_audio.device != device:
            raise ValueError("semantic_audio must be on the target device")
        if semantic_audio.requires_grad:
            raise RuntimeError("semantic_audio must not require gradients")

        semantic_attention_mask = (
            torch.arange(max_time, device=device).unsqueeze(0) < output_lengths.unsqueeze(1)
        )
        if spatial_parameter is None:
            spatial_encoder_input = spatial_features.to(device=device)
        else:
            spatial_encoder_input = spatial_features.to(
                device=spatial_parameter.device, dtype=spatial_parameter.dtype
            )
        spatial_seq = self.spatial_encoder(spatial_encoder_input)
        self._validate_spatial_sequence(
            spatial_seq,
            batch_size=mel_features.shape[0],
            device=device,
        )
        spatial_seq = spatial_seq.to(dtype=torch.float32)
        fused = self.fusion._forward_validated(
            semantic_audio, spatial_seq, semantic_attention_mask
        )
        if not torch.equal(fused["audio_content_mask"], semantic_attention_mask):
            raise RuntimeError("fusion audio_content_mask must match semantic_attention_mask")
        return {
            **fused,
            "feature_attention_mask": mel_mask.to(dtype=torch.long),
            "semantic_attention_mask": semantic_attention_mask,
            "output_lengths": output_lengths,
        }

    @staticmethod
    def _normalize_target_device(device: torch.device) -> torch.device:
        normalized = torch.device(device)
        if normalized.type != "cuda":
            return normalized
        if not torch.cuda.is_available():
            raise ValueError(f"target device {normalized} invalid: CUDA unavailable")
        index = normalized.index
        if index is None:
            index = torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"target device cuda:{index} is not a visible CUDA device"
            )
        return torch.device("cuda", index)

    @staticmethod
    def _validate_whisper_attention_mask(
        mask: object,
        batch_size: int,
        time: int,
        source_device: torch.device,
    ) -> None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("whisper_attention_mask must be a torch.Tensor")
        if tuple(mask.shape) != (batch_size, time):
            raise ValueError("whisper_attention_mask must have shape [batch, time]")
        if mask.layout != torch.strided:
            raise ValueError("whisper_attention_mask must use strided layout")
        if mask.device.type == "meta":
            raise ValueError("whisper_attention_mask must not use the meta device")
        if mask.is_quantized:
            raise ValueError("whisper_attention_mask must not be quantized")
        if mask.device != source_device:
            raise ValueError(
                "whisper_attention_mask must share the same device as "
                "whisper_input_features"
            )
        if mask.dtype != torch.bool:
            try:
                torch.iinfo(mask.dtype)
            except TypeError as error:
                raise ValueError(
                    "whisper_attention_mask dtype must be bool or integer"
                ) from error
        valid = mask == 1
        if not torch.all((mask == 0) | valid):
            raise ValueError("whisper_attention_mask must be binary")
        if not torch.all(torch.any(valid, dim=1)):
            raise ValueError(
                "whisper_attention_mask must contain at least one valid frame per sample"
            )
        if time > 1 and torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("whisper_attention_mask must be left-aligned")

    @staticmethod
    def _validate_floating_sequence(
        value: object, name: str, shape_description: str
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.layout != torch.strided:
            raise TypeError(f"{name} must use strided layout")
        if value.device.type == "meta":
            raise ValueError(f"{name} must not use the meta device")
        if value.is_quantized:
            raise TypeError(f"{name} must not be quantized")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if value.ndim != 3:
            raise ValueError(f"{name} must have shape {shape_description}")
        if any(size == 0 for size in value.shape):
            raise ValueError(f"{name} must be non-empty")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")

    def _validate_spatial_sequence(
        self, spatial_seq: object, batch_size: int, device: torch.device
    ) -> None:
        if not isinstance(spatial_seq, torch.Tensor):
            raise TypeError("spatial_seq must be a torch.Tensor")
        self._validate_floating_sequence(
            spatial_seq, "spatial_seq", "[batch, time, hidden]"
        )
        if spatial_seq.shape[0] != batch_size:
            raise ValueError("spatial_seq batch size must match whisper_input_features")
        if spatial_seq.shape[2] != self.fusion.spatial_dim:
            raise ValueError("spatial_seq hidden dimension must match fusion.spatial_dim")
        if spatial_seq.device != device:
            raise ValueError("spatial_seq must be on the target device")

    @staticmethod
    def _validate_batch_fields(
        batch: object, field_names: Sequence[str]
    ) -> int:
        """Validate training metadata before any audio encoding work."""
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        missing = [name for name in field_names if name not in batch]
        if missing:
            raise ValueError("batch is missing required field(s): " + ", ".join(missing))

        tensor_fields = {
            "whisper_input_features",
            "spatial_features",
            "whisper_attention_mask",
        }
        sequence_lengths: list[int] = []
        for name in field_names:
            if name in tensor_fields:
                continue
            value = batch[name]
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{name} must be a non-string sequence")
            sequence_lengths.append(len(value))
        if not sequence_lengths or sequence_lengths[0] == 0:
            raise ValueError("training batch sequences must have non-zero length")
        batch_size = sequence_lengths[0]
        if any(length != batch_size for length in sequence_lengths):
            raise ValueError("training batch sequence fields must have the same length")

        for name in ("whisper_input_features", "spatial_features"):
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"{name} batch axis must match training batch length")
        if "whisper_attention_mask" in batch:
            mask = batch["whisper_attention_mask"]
            if not isinstance(mask, torch.Tensor):
                raise TypeError("whisper_attention_mask must be a torch.Tensor")
            if mask.ndim == 0 or mask.shape[0] != batch_size:
                raise ValueError(
                    "whisper_attention_mask batch axis must match training batch length"
                )
        return batch_size

    @staticmethod
    def _format_mcq_targets(
        correct_options: Sequence[object],
        options_batch: Sequence[object],
        answer_texts: Sequence[object],
    ) -> List[str]:
        lengths = (len(correct_options), len(options_batch), len(answer_texts))
        if len(set(lengths)) != 1:
            raise ValueError("MCQ target fields must have the same length")
        return [
            format_mcq_target(correct, options, answer)
            for correct, options, answer in zip(correct_options, options_batch, answer_texts)
        ]

    def _prepare_training_inputs(
        self,
        messages_batch: Sequence[Sequence[object]],
        answers: Sequence[str],
        fused: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> Mapping[str, torch.Tensor | None]:
        if isinstance(messages_batch, (str, bytes)) or not isinstance(messages_batch, Sequence):
            raise TypeError("messages_batch must be a non-string sequence")
        if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
            raise TypeError("answers must be a non-string sequence")
        if not isinstance(fused, Mapping):
            raise TypeError("fused must be a mapping")
        required = ("fused_audio", "output_lengths", "feature_attention_mask")
        missing = [name for name in required if name not in fused]
        if missing:
            raise ValueError("fused is missing required field(s): " + ", ".join(missing))
        batch_size = len(messages_batch)
        if batch_size == 0 or len(answers) != batch_size:
            raise ValueError("messages_batch and answers must have the same non-zero length")
        for name in required:
            value = fused[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"fused {name} must be a torch.Tensor")
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"fused {name} batch axis must match messages_batch")

        prompt_ids_list: list[torch.Tensor] = []
        answer_ids_list: list[torch.Tensor] = []
        for sample_index, (messages, answer) in enumerate(zip(messages_batch, answers)):
            prompt_ids, answer_ids = self._render_training_ids(
                messages,
                answer,
                audio_output_length=int(fused["output_lengths"][sample_index]),
                sample_index=sample_index,
            )
            prompt_ids_list.append(prompt_ids)
            answer_ids_list.append(answer_ids)
        model_inputs = self.backbone.build_lm_inputs(
            prompt_ids_list=prompt_ids_list,
            audio_tokens=fused["fused_audio"],
            feature_attention_mask=fused["feature_attention_mask"],
            device=device,
            answer_ids_list=answer_ids_list,
        )
        if not isinstance(model_inputs, Mapping):
            raise TypeError("build_lm_inputs must return a mapping")
        expected = (
            "input_ids", "inputs_embeds", "attention_mask", "feature_attention_mask",
            "labels", "input_features",
        )
        missing = [name for name in expected if name not in model_inputs]
        if missing:
            raise ValueError("build_lm_inputs output is missing: " + ", ".join(missing))
        if model_inputs["input_features"] is not None:
            raise ValueError("build_lm_inputs input_features must be None")
        return model_inputs

    @staticmethod
    def _validate_generation_model_inputs(model_inputs: object) -> None:
        if not isinstance(model_inputs, Mapping):
            raise TypeError("build_lm_inputs must return a mapping")
        expected = (
            "input_ids", "inputs_embeds", "attention_mask",
            "feature_attention_mask", "input_features",
        )
        missing = [name for name in expected if name not in model_inputs]
        if missing:
            raise ValueError("build_lm_inputs output is missing: " + ", ".join(missing))
        if model_inputs["input_features"] is not None:
            raise ValueError("build_lm_inputs input_features must be None")

    @torch.no_grad()
    def _generate_texts(
        self,
        messages_batch: Sequence[Sequence[object]],
        fused: Mapping[str, torch.Tensor],
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> list[str]:
        max_new_tokens = self._validate_positive_integer(
            max_new_tokens, "max_new_tokens"
        )
        num_beams = self._validate_positive_integer(num_beams, "num_beams")
        if isinstance(messages_batch, (str, bytes)) or not isinstance(messages_batch, Sequence):
            raise TypeError("messages_batch must be a non-string sequence")
        if not messages_batch:
            raise ValueError("messages_batch must be non-empty")
        if not isinstance(fused, Mapping):
            raise TypeError("fused must be a mapping")
        required = ("fused_audio", "output_lengths", "feature_attention_mask")
        missing = [name for name in required if name not in fused]
        if missing:
            raise ValueError("fused is missing required field(s): " + ", ".join(missing))
        batch_size = len(messages_batch)
        for name in required:
            value = fused[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"fused {name} must be a torch.Tensor")
            if value.layout != torch.strided or value.device.type == "meta" or value.is_quantized:
                raise TypeError(f"fused {name} must be a strided, non-meta, non-quantized Tensor")
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"fused {name} batch axis must match messages_batch")
        output_lengths = fused["output_lengths"]
        if output_lengths.ndim != 1:
            raise ValueError("fused output_lengths must be one-dimensional")
        output_lengths_list = output_lengths.detach().to(device="cpu").tolist()
        try:
            output_lengths_list = [
                self._validate_positive_integer(length, "audio_output_length")
                for length in output_lengths_list
            ]
        except (TypeError, ValueError) as error:
            raise type(error)(f"fused output_lengths: {error}") from error

        prompt_ids_list: list[torch.Tensor] = []
        for messages, output_length in zip(messages_batch, output_lengths_list):
            if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
                raise TypeError("messages_batch entries must be non-string sequences")
            prompt_ids_list.append(self._render_prompt_ids(messages, output_length))
        model_inputs = self.backbone.build_lm_inputs(
            prompt_ids_list=prompt_ids_list,
            audio_tokens=fused["fused_audio"],
            feature_attention_mask=fused["feature_attention_mask"],
            device=device,
            answer_ids_list=None,
            left_pad=True,
        )
        self._validate_generation_model_inputs(model_inputs)
        generated = self.backbone.generate_from_inputs_embeds(
            model_inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
        )
        if isinstance(generated, (str, bytes)) or not isinstance(generated, Sequence):
            raise TypeError("generated texts must be a non-string sequence")
        if len(generated) != batch_size:
            raise ValueError("generated texts batch size must match messages_batch")
        if not all(isinstance(text, str) for text in generated):
            raise TypeError("generated text items must be strings")
        return [text.strip() for text in generated]

    @torch.no_grad()
    def _generate_dual_layer_texts(
        self,
        prebuilt_layer1_messages: Sequence[Sequence[object]],
        normalized_layer2_inputs: Sequence[tuple[str, dict[str, str]]],
        fused: Mapping[str, torch.Tensor],
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> tuple[list[str], list[str]]:
        layer1_text = self._generate_texts(
            prebuilt_layer1_messages, fused, device, max_new_tokens, num_beams
        )
        if len(normalized_layer2_inputs) != len(layer1_text):
            raise ValueError("normalized_layer2_inputs must match layer1 batch size")
        layer2_messages = [
            self._extend_layer1_messages_from_normalized(
                layer1_messages,
                self._normalize_history_text(prediction),
                layer2_question,
                layer2_options,
            )
            for layer1_messages, prediction, (layer2_question, layer2_options) in zip(
                prebuilt_layer1_messages, layer1_text, normalized_layer2_inputs
            )
        ]
        layer2_text = self._generate_texts(
            layer2_messages, fused, device, max_new_tokens, num_beams
        )
        return layer1_text, layer2_text

    @torch.no_grad()
    def generate_freeform_answers(
        self,
        batch: Mapping[str, object],
        device: torch.device,
        max_new_tokens: int = 32,
        num_beams: int = 1,
    ) -> Qwen3OmniFreeformGenerationOutput:
        max_new_tokens = self._validate_positive_integer(
            max_new_tokens, "max_new_tokens"
        )
        num_beams = self._validate_positive_integer(num_beams, "num_beams")
        fields = (
            "whisper_input_features", "spatial_features", "layer1_questions",
            "layer1_options", "layer2_questions", "layer2_options",
        )
        batch_size = self._validate_batch_fields(batch, fields)
        layer1_messages = [
            self._build_layer1_messages(question, options)
            for question, options in zip(batch["layer1_questions"], batch["layer1_options"])
        ]
        normalized_layer2_inputs = [
            self._normalize_mcq_inputs(question, options)
            for question, options in zip(batch["layer2_questions"], batch["layer2_options"])
        ]

        with self._state_lock():
            was_training = self.training
            try:
                if was_training:
                    self.eval()
                fused = self._encode_fused_audio(batch, device)
                layer1_text, layer2_text = self._generate_dual_layer_texts(
                    layer1_messages,
                    normalized_layer2_inputs,
                    fused,
                    device,
                    max_new_tokens,
                    num_beams,
                )
                return {
                    "layer1_text": layer1_text,
                    "layer2_text": layer2_text,
                    "layer1_gate": [1.0] * batch_size,
                    "layer2_gate": [1.0] * batch_size,
                }
            finally:
                if self.training != was_training:
                    self.train(was_training)

    def _layer_lm_loss(
        self, model_inputs: Mapping[str, torch.Tensor | None]
    ) -> tuple[torch.Tensor, int]:
        self._validate_training_model_inputs(model_inputs)
        labels = model_inputs["labels"]
        token_count = int((labels != -100).sum().item())
        if token_count == 0:
            raise ValueError("layer training requires at least one supervised token")
        output = self.backbone.forward_lm(model_inputs)
        loss = getattr(output, "loss", None)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("forward_lm output loss must be a torch.Tensor")
        if loss.ndim != 0:
            raise ValueError("forward_lm output loss must be scalar")
        if not loss.is_floating_point():
            raise TypeError("forward_lm output loss must have a floating-point dtype")
        if loss.layout != torch.strided:
            raise TypeError("forward_lm output loss must use strided layout")
        if loss.device.type == "meta":
            raise ValueError("forward_lm output loss must not use the meta device")
        if loss.is_quantized:
            raise TypeError("forward_lm output loss must not be quantized")
        if loss.device != model_inputs["inputs_embeds"].device:
            raise ValueError("forward_lm output loss must share the inputs_embeds device")
        if not torch.isfinite(loss):
            raise ValueError("forward_lm output loss must be finite")
        if torch.is_grad_enabled() and not loss.requires_grad:
            raise ValueError("forward_lm output loss must require gradients when grad is enabled")
        return loss, token_count

    @staticmethod
    def _validate_training_model_inputs(model_inputs: object) -> None:
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        names = ("input_ids", "inputs_embeds", "attention_mask", "feature_attention_mask", "labels", "input_features")
        missing = [name for name in names if name not in model_inputs]
        if missing:
            raise ValueError("model_inputs missing: " + ", ".join(missing))
        if model_inputs["input_features"] is not None:
            raise ValueError("model_inputs input_features must be None")
        def tensor(name: str) -> torch.Tensor:
            value = model_inputs[name]
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device.type == "meta" or value.is_quantized:
                raise TypeError(f"model_inputs {name} must be a strided, non-meta, non-quantized Tensor")
            return value
        ids, embeds, attention, feature, labels = (tensor(name) for name in names[:5])
        if labels.dtype != torch.long or labels.ndim != 2 or not all(labels.shape):
            raise ValueError("model_inputs labels must be a non-empty 2D torch.long Tensor")
        if ids.ndim != 2 or ids.dtype == torch.bool or ids.shape != labels.shape or ids.device != labels.device:
            raise ValueError("model_inputs input_ids must match labels shape/device and use a non-bool integer dtype")
        try:
            torch.iinfo(ids.dtype)
        except TypeError as error:
            raise TypeError(
                "model_inputs input_ids must be integer"
            ) from error
        if embeds.ndim != 3 or tuple(embeds.shape[:2]) != tuple(labels.shape) or not embeds.is_floating_point() or embeds.device != labels.device or not torch.isfinite(embeds).all():
            raise ValueError("model_inputs inputs_embeds must be finite floating [batch, sequence, hidden] on labels device")
        for name, value in (("attention_mask", attention), ("feature_attention_mask", feature)):
            if value.ndim != 2 or value.shape[0] != labels.shape[0] or value.device != labels.device or not torch.all((value == 0) | (value == 1)):
                raise ValueError(f"model_inputs {name} must be binary 2D on the input device")

    def forward_train(
        self,
        batch: Mapping[str, object],
        device: torch.device,
        enable_layer2: bool = True,
    ) -> Dict[str, torch.Tensor]:
        with self._state_lock():
            return self._forward_train_locked(batch, device, enable_layer2)

    def _forward_train_locked(
        self,
        batch: Mapping[str, object],
        device: torch.device,
        enable_layer2: bool,
    ) -> Dict[str, torch.Tensor]:
        if type(enable_layer2) is not bool:
            raise TypeError("enable_layer2 must be bool")
        layer1_fields = (
            "whisper_input_features", "spatial_features", "layer1_questions",
            "layer1_options", "layer1_correct_option", "layer1_answers",
        )
        layer2_fields = (
            "layer2_questions", "layer2_options", "layer2_correct_option", "layer2_answers",
        )
        self._validate_batch_fields(batch, layer1_fields + (layer2_fields if enable_layer2 else ()))
        if not enable_layer2 and self.loss_weights["lm_layer1"] <= 0:
            raise ValueError("lm_layer1 must be greater than zero when layer2 is disabled")

        layer1_samples = [
            self._normalize_training_mcq_sample(question, options, correct_option)
            for question, options, correct_option in zip(
                batch["layer1_questions"],
                batch["layer1_options"],
                batch["layer1_correct_option"],
            )
        ]
        layer1_targets = self._format_mcq_targets(
            [sample[2] for sample in layer1_samples],
            [sample[1] for sample in layer1_samples],
            batch["layer1_answers"],
        )
        layer2_samples: list[tuple[str, dict[str, str], str]] | None = None
        layer2_targets: list[str] | None = None
        if enable_layer2:
            layer2_samples = [
                self._normalize_training_mcq_sample(question, options, correct_option)
                for question, options, correct_option in zip(
                    batch["layer2_questions"],
                    batch["layer2_options"],
                    batch["layer2_correct_option"],
                )
            ]
            layer2_targets = self._format_mcq_targets(
                [sample[2] for sample in layer2_samples],
                [sample[1] for sample in layer2_samples],
                batch["layer2_answers"],
            )
        layer1_messages = [
            self._build_layer1_messages_from_normalized(question, options)
            for question, options, _ in layer1_samples
        ]
        layer2_messages: list[list[object]] | None = None
        if enable_layer2:
            layer2_messages = [
                self._extend_layer1_messages_from_normalized(
                    layer1_message, layer1_target, layer2_question, layer2_options
                )
                for layer1_message, layer1_target, (
                    layer2_question, layer2_options, _,
                ) in zip(layer1_messages, layer1_targets, layer2_samples)
            ]
        device = self._normalize_target_device(device)
        fused = self._encode_fused_audio(batch, device)
        layer1_inputs = self._prepare_training_inputs(layer1_messages, layer1_targets, fused, device)
        loss1, tokens1 = self._layer_lm_loss(layer1_inputs)
        gate1 = fused["gate"].mean()

        if enable_layer2:
            layer2_inputs = self._prepare_training_inputs(layer2_messages, layer2_targets, fused, device)
            loss2, tokens2 = self._layer_lm_loss(layer2_inputs)
            gate2 = fused["gate"].mean()
            max_weight = max(self.loss_weights.values())
            scaled1 = self.loss_weights["lm_layer1"] / max_weight
            scaled2 = self.loss_weights["lm_layer2"] / max_weight
            mass1, mass2 = tokens1 * scaled1, tokens2 * scaled2
            denominator = mass1 + mass2
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError("training loss denominator must be finite and positive")
            loss = (
                loss1 * (mass1 / denominator) + loss2 * (mass2 / denominator)
            )
        else:
            loss2 = torch.zeros((), device=loss1.device, dtype=loss1.dtype)
            gate2 = torch.zeros((), device=loss1.device, dtype=loss1.dtype)
            loss = loss1

        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not loss.is_floating_point()
            or loss.device != torch.device(device)
            or not torch.isfinite(loss)
        ):
            raise ValueError("training loss must be a finite scalar floating Tensor on the target device")

        return {
            "loss": loss,
            "loss_next_token": loss.detach(),
            "loss_layer1_next_token": loss1.detach(),
            "loss_layer2_next_token": loss2.detach(),
            "mean_gate_layer1": gate1.detach(),
            "mean_gate_layer2": gate2.detach(),
        }
