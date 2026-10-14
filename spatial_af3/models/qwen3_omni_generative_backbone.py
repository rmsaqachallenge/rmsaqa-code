from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
import re
import threading
import weakref
from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from transformers.utils import is_flash_attn_2_available

from ..layered_mcq import build_mcq_prompt_body


_AUDIO_CAPTURE_INVOCATION: ContextVar[object | None] = ContextVar(
    "qwen3_omni_audio_capture_invocation",
    default=None,
)
_AUDIO_CAPTURE_DEPTH: ContextVar[int] = ContextVar(
    "qwen3_omni_audio_capture_depth",
    default=0,
)
_STATE_LOCKS_GUARD = threading.Lock()
_STATE_LOCKS: weakref.WeakKeyDictionary[
    nn.Module,
    threading.RLock,
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class FeatureEquivalenceReport:
    mel_shape_equal: bool
    mask_equal: bool
    mel_max_abs: float
    mel_mean_abs: float
    mel_allclose: bool
    semantic_shape_equal: bool
    semantic_max_abs: float
    semantic_mean_abs: float
    semantic_allclose: bool
    passed: bool


class _PreProjectionCaptured(RuntimeError):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__("captured Audio Tower ln_post output")
        self.hidden_states = hidden_states


class FrozenQwen3OmniGenerativeBackbone(nn.Module):
    """Frozen Qwen3-Omni semantic backbone with trainable Thinker LoRA."""

    LORA_ADAPTER_NAME = "default"

    # The official checkpoint uses vocab.json + merges.txt and omits the
    # optional processor_config.json/tokenizer.json files.
    REQUIRED_FILES = (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    )
    WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")

    def __init__(
        self,
        qwen3_omni_root: str | Path,
        text_dtype: torch.dtype = torch.bfloat16,
        audio_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        enable_llm_lora: bool = True,
        llm_lora_r: int = 12,
        llm_lora_alpha: int = 12,
        llm_lora_dropout: float = 0.05,
        llm_lora_num_layers: int = 48,
    ) -> None:
        super().__init__()
        self.qwen3_omni_root = self._validate_checkpoint_root(qwen3_omni_root)
        self.text_dtype = text_dtype
        self.audio_dtype = audio_dtype
        self.attn_implementation = attn_implementation
        self.enable_llm_lora = bool(enable_llm_lora)
        self.llm_lora_r = int(llm_lora_r)
        self.llm_lora_alpha = int(llm_lora_alpha)
        self.llm_lora_dropout = float(llm_lora_dropout)
        self.llm_lora_num_layers = int(llm_lora_num_layers)

        if (
            attn_implementation == "flash_attention_2"
            and not is_flash_attn_2_available()
        ):
            raise ImportError(
                "attn_implementation='flash_attention_2' requires "
                "flash-attn 2.8.3. Install it on Linux with "
                "`pip install flash-attn==2.8.3 --no-build-isolation`, or "
                "select an explicit non-FlashAttention implementation such "
                "as 'eager'."
            )

        config = Qwen3OmniMoeConfig.from_pretrained(
            self.qwen3_omni_root,
            local_files_only=True,
        )
        config.enable_audio_output = False
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.qwen3_omni_root,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.llm = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.qwen3_omni_root,
            config=config,
            local_files_only=True,
            dtype=text_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
        self.llm.disable_talker()
        self.audio_encoder.to(dtype=audio_dtype)

        self.hidden_size = int(self.thinker.config.text_config.hidden_size)
        self.audio_hidden_size = int(self.thinker.config.audio_config.d_model)
        self.num_mel_bins = int(self.thinker.config.audio_config.num_mel_bins)
        self.audio_token_id = self._validated_token_id(
            self.thinker.config.audio_token_id,
            "Thinker audio token id",
        )
        self.pad_token_id = self._validated_token_id(
            self.tokenizer.pad_token_id,
            "tokenizer pad token id",
        )
        self.eos_token_id = self._validated_token_id(
            self.tokenizer.eos_token_id,
            "tokenizer eos token id",
        )

        audio_token = self.processor.audio_token
        if not isinstance(audio_token, str) or not audio_token:
            raise ValueError(
                "processor.audio_token must be a non-empty string"
            )
        tokenizer_vocab = self.tokenizer.get_vocab()
        if audio_token not in tokenizer_vocab:
            raise ValueError(
                f"audio token {audio_token!r} is missing from tokenizer vocabulary"
            )
        vocab_audio_id = self._validated_token_id(
            tokenizer_vocab[audio_token],
            "tokenizer vocabulary audio token id",
        )
        converted_audio_id = self._validated_token_id(
            self.tokenizer.convert_tokens_to_ids(audio_token),
            "converted audio token id",
        )
        if not (
            vocab_audio_id == converted_audio_id == self.audio_token_id
        ):
            raise ValueError(
                "audio token ids disagree: tokenizer vocabulary has "
                f"{vocab_audio_id}, token conversion has {converted_audio_id}, "
                f"and Thinker config has {self.audio_token_id}"
            )

        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        if self.enable_llm_lora:
            self._validate_lora_config()
            self._inject_trainable_llm_lora()
        self.thinker.config.use_cache = False
        nested_cache_configs = (
            getattr(self.thinker.config, "text_config", None),
            getattr(self.thinker.model, "config", None),
        )
        for cache_config in nested_cache_configs:
            if cache_config is not None and hasattr(cache_config, "use_cache"):
                cache_config.use_cache = False
        self.thinker.gradient_checkpointing_enable()
        self.freeze()
        self.train(False)

    @property
    def thinker(self) -> nn.Module:
        return self.llm.thinker

    @property
    def audio_encoder(self) -> nn.Module:
        return self.thinker.audio_tower

    @property
    def vision_encoder(self) -> nn.Module:
        return self.thinker.visual

    @classmethod
    def _validate_checkpoint_root(cls, value: str | Path) -> Path:
        root = Path(value)
        if not root.is_dir():
            raise FileNotFoundError(
                f"Qwen3-Omni checkpoint directory does not exist: {root}"
            )
        missing = [
            name for name in cls.REQUIRED_FILES if not (root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Qwen3-Omni checkpoint is missing required files: "
                + ", ".join(missing)
            )
        if not any((root / name).is_file() for name in cls.WEIGHT_FILES):
            raise FileNotFoundError(
                "Qwen3-Omni checkpoint requires model.safetensors or "
                "model.safetensors.index.json"
            )
        return root

    def _validate_lora_config(self) -> None:
        if self.llm_lora_r <= 0 or self.llm_lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.llm_lora_num_layers <= 0:
            raise ValueError("llm_lora_num_layers must be positive")
        if not 0.0 <= self.llm_lora_dropout < 1.0:
            raise ValueError("llm_lora_dropout must be in [0, 1)")

    def _inject_trainable_llm_lora(self) -> None:
        actual_num_layers = int(
            self.thinker.config.text_config.num_hidden_layers
        )
        if self.llm_lora_num_layers > actual_num_layers:
            raise ValueError(
                f"requested {self.llm_lora_num_layers} LoRA layers, but "
                f"checkpoint has {actual_num_layers}"
            )

        layer_indexes = "|".join(
            str(layer_index)
            for layer_index in range(self.llm_lora_num_layers)
        )
        target_modules = (
            rf"^layers\.(?:{layer_indexes})\.self_attn\."
            r"(?:q_proj|k_proj|v_proj|o_proj)$"
        )
        config = LoraConfig(
            r=self.llm_lora_r,
            lora_alpha=self.llm_lora_alpha,
            lora_dropout=self.llm_lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules,
        )
        self.thinker.model = get_peft_model(self.thinker.model, config)
        self.audit_lora_targets()

    def audit_lora_targets(self) -> dict[int, set[str]]:
        projection_pattern = re.compile(
            r"(?:^|\.)layers\.(\d+)\.self_attn\."
            r"(q_proj|k_proj|v_proj|o_proj)$"
        )
        expected = {
            (layer_index, projection)
            for layer_index in range(self.llm_lora_num_layers)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        wrapper_names: dict[tuple[int, str], list[str]] = {}
        unrecognized: set[str] = set()
        configuration_errors: set[str] = set()

        for name, module in self._named_modules_including_aliases(
            self.thinker.model
        ):
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            match = projection_pattern.search(name)
            if match is None:
                unrecognized.add(name)
                continue
            target = (int(match.group(1)), match.group(2))
            wrapper_names.setdefault(target, []).append(name)
            self._audit_lora_adapter_configuration(
                name,
                module,
                configuration_errors,
            )

        found = set(wrapper_names)

        missing_names = {
            f"layers.{layer_index}.self_attn.{projection}"
            for layer_index, projection in expected - found
        }
        unexpected_names = {
            f"layers.{layer_index}.self_attn.{projection}"
            for layer_index, projection in found - expected
        } | unrecognized | {
            name
            for names in wrapper_names.values()
            if len(names) > 1
            for name in names
        }
        if missing_names or unexpected_names or configuration_errors:
            raise RuntimeError(
                "LoRA target audit failed: "
                f"missing={sorted(missing_names)}; "
                f"unexpected={sorted(unexpected_names)}; "
                f"adapter configuration={sorted(configuration_errors)}"
            )

        audited: dict[int, set[str]] = {
            layer_index: set()
            for layer_index in range(self.llm_lora_num_layers)
        }
        for layer_index, projection in found:
            audited[layer_index].add(projection)
        return audited

    @staticmethod
    def _named_modules_including_aliases(
        module: nn.Module,
    ) -> Iterator[tuple[str, nn.Module]]:
        try:
            named_modules = module.named_modules(remove_duplicate=False)
        except TypeError:
            pass
        else:
            yield from named_modules
            return

        def walk(
            current: nn.Module,
            prefix: str,
            ancestors: frozenset[int],
        ) -> Iterator[tuple[str, nn.Module]]:
            yield prefix, current
            next_ancestors = ancestors | {id(current)}
            for child_name, child in current._modules.items():
                if child is None:
                    continue
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                if id(child) in next_ancestors:
                    yield child_prefix, child
                    continue
                yield from walk(child, child_prefix, next_ancestors)

        yield from walk(module, "", frozenset())

    def _audit_lora_adapter_configuration(
        self,
        module_name: str,
        module: nn.Module,
        errors: set[str],
    ) -> None:
        expected_adapters = {self.LORA_ADAPTER_NAME}
        adapter_collections = {
            "lora_A": module.lora_A,
            "lora_B": module.lora_B,
            "lora_dropout": module.lora_dropout,
            "rank": module.r,
            "alpha": module.lora_alpha,
            "scaling": module.scaling,
        }
        for label, collection in adapter_collections.items():
            adapters = set(collection.keys())
            if adapters != expected_adapters:
                errors.add(
                    f"{module_name}: unexpected adapter set in {label}: "
                    f"{sorted(adapters)}"
                )

        adapter_name = self.LORA_ADAPTER_NAME
        if any(
            adapter_name not in collection
            for collection in adapter_collections.values()
        ):
            return
        if module.r[adapter_name] != self.llm_lora_r:
            errors.add(
                f"{module_name}: rank is {module.r[adapter_name]}, "
                f"expected {self.llm_lora_r}"
            )
        if module.lora_alpha[adapter_name] != self.llm_lora_alpha:
            errors.add(
                f"{module_name}: alpha is {module.lora_alpha[adapter_name]}, "
                f"expected {self.llm_lora_alpha}"
            )

        dropout = module.lora_dropout[adapter_name]
        if isinstance(dropout, nn.Identity):
            actual_dropout = 0.0
        else:
            actual_dropout = getattr(dropout, "p", None)
        if actual_dropout != self.llm_lora_dropout:
            errors.add(
                f"{module_name}: dropout is {actual_dropout}, "
                f"expected {self.llm_lora_dropout}"
            )

        expected_scaling = self.llm_lora_alpha / self.llm_lora_r
        if module.scaling[adapter_name] != expected_scaling:
            errors.add(
                f"{module_name}: scaling is {module.scaling[adapter_name]}, "
                f"expected {expected_scaling}"
            )

    @staticmethod
    def _validated_token_id(value: object, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a non-negative integer")
        try:
            token_id = index(value)
        except TypeError as error:
            raise ValueError(
                f"{label} must be a non-negative integer"
            ) from error
        if token_id < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return int(token_id)

    def build_user_messages(
        self,
        question: str,
        options: Optional[Dict[str, str]] = None,
    ) -> list[dict[str, object]]:
        question = question.strip()
        if not question:
            raise ValueError("question must be non-empty after stripping")
        prefix = (
            "You are given a first-order ambisonics audio clip. Use the audio "
            "content faithfully, and use spatial cues only when the question "
            "requires them.\n"
        )
        if options is None:
            body = f"Question: {question}\nAnswer briefly and directly."
        else:
            if not isinstance(options, dict) or not options:
                raise ValueError("options must be a non-empty dict")
            normalized_options: Dict[str, str] = {}
            for raw_key, raw_text in options.items():
                key = str(raw_key).strip().upper()
                if not key:
                    raise ValueError(
                        "each option key must be non-empty after stripping"
                    )
                if key in normalized_options:
                    raise ValueError(
                        f"duplicate option key after normalization: {key}"
                    )
                text = str(raw_text).strip()
                if not text:
                    raise ValueError(
                        f"option text for {key} must be non-empty after stripping"
                    )
                normalized_options[key] = text
            body = build_mcq_prompt_body(
                question=question,
                options=normalized_options,
            )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "cached://foa-w-channel"},
                    {"type": "text", "text": prefix + body},
                ],
            }
        ]

    def build_chat_prompt_ids_from_messages(
        self,
        messages: Sequence[Dict[str, object]],
        audio_output_length: int,
        add_generation_prompt: bool = True,
    ) -> torch.Tensor:
        if isinstance(audio_output_length, bool):
            raise ValueError("audio_output_length must be a positive integer")
        try:
            audio_output_length = index(audio_output_length)
        except TypeError as error:
            raise ValueError(
                "audio_output_length must be a positive integer"
            ) from error
        if audio_output_length <= 0:
            raise ValueError("audio_output_length must be a positive integer")

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        audio_token = self.processor.audio_token
        placeholder_count = prompt_text.count(audio_token)
        if placeholder_count != 1:
            raise ValueError(
                "rendered chat template must contain exactly one unexpanded "
                f"audio token, got {placeholder_count}"
            )
        prompt_text = prompt_text.replace(
            audio_token,
            audio_token * audio_output_length,
            1,
        )
        tokenizer_output = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        if not hasattr(tokenizer_output, "input_ids"):
            raise ValueError("tokenizer output must define input_ids")
        tokenized_input_ids = tokenizer_output.input_ids
        if not isinstance(tokenized_input_ids, torch.Tensor):
            raise ValueError("tokenizer input_ids must be a torch.Tensor")
        if (
            tokenized_input_ids.ndim != 2
            or tokenized_input_ids.shape[0] != 1
            or tokenized_input_ids.shape[1] == 0
        ):
            raise ValueError(
                "tokenizer input_ids must have non-empty shape [1, length], "
                f"got {tuple(tokenized_input_ids.shape)}"
            )
        try:
            torch.iinfo(tokenized_input_ids.dtype)
        except TypeError as error:
            raise ValueError(
                "tokenizer input_ids must use a PyTorch integer dtype, "
                f"got {tokenized_input_ids.dtype}"
            ) from error
        prompt_ids = tokenized_input_ids[0].to(dtype=torch.long)
        audio_positions = torch.where(prompt_ids == self.audio_token_id)[0]
        actual_count = int(audio_positions.numel())
        if actual_count != audio_output_length:
            raise ValueError(
                "tokenizer produced "
                f"{actual_count} audio token ids, expected {audio_output_length}"
            )
        expected_positions = torch.arange(
            audio_positions[0],
            audio_positions[0] + audio_output_length,
            device=audio_positions.device,
        )
        if not torch.equal(audio_positions, expected_positions):
            raise ValueError(
                "tokenizer-produced audio token ids must form one adjacent span"
            )
        return prompt_ids

    def get_input_embeddings(self) -> nn.Module:
        return self.thinker.get_input_embeddings()

    def _state_lock(self) -> threading.RLock:
        with _STATE_LOCKS_GUARD:
            lock = _STATE_LOCKS.get(self)
            if lock is None:
                lock = threading.RLock()
                _STATE_LOCKS[self] = lock
            return lock

    def _set_training_mode_without_freeze(self, mode: bool) -> None:
        nn.Module.train(self, mode)
        self.audio_encoder.eval()
        self.vision_encoder.eval()
        self.thinker.lm_head.eval()
        self.thinker.model.train(mode and self.enable_llm_lora)

    @staticmethod
    def _require_integer_tensor_dtype(
        tensor: torch.Tensor,
        field_name: str,
    ) -> None:
        if tensor.dtype == torch.bool:
            raise ValueError(
                f"{field_name} must use an integer dtype, not torch.bool"
            )
        try:
            torch.iinfo(tensor.dtype)
        except TypeError as error:
            raise ValueError(
                f"{field_name} must use an integer dtype, got {tensor.dtype}"
            ) from error

    @staticmethod
    def _require_binary_mask_dtype(
        tensor: torch.Tensor,
        field_name: str,
    ) -> None:
        if tensor.is_quantized:
            raise ValueError(
                f"{field_name} must use a bool, integer, or floating dtype, "
                f"got quantized dtype {tensor.dtype}"
            )
        if tensor.dtype == torch.bool or torch.is_floating_point(tensor):
            return
        try:
            torch.iinfo(tensor.dtype)
        except TypeError as error:
            raise ValueError(
                f"{field_name} must use a bool, integer, or floating dtype, "
                f"got {tensor.dtype}"
            ) from error

    def _validate_lm_model_inputs(
        self,
        model_inputs: Dict[str, torch.Tensor | None],
        *,
        require_labels: bool,
    ) -> None:
        required_keys = [
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "feature_attention_mask",
        ]
        if require_labels:
            required_keys.append("labels")
        missing = [key for key in required_keys if key not in model_inputs]
        if missing:
            raise KeyError(
                "missing required model_inputs keys: " + ", ".join(missing)
            )

        input_ids = model_inputs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if batch_size == 0 or sequence_length == 0:
            raise ValueError(
                "input_ids must have non-empty batch and sequence axes"
            )
        self._require_integer_tensor_dtype(input_ids, "input_ids")

        inputs_embeds = model_inputs["inputs_embeds"]
        if not isinstance(inputs_embeds, torch.Tensor):
            raise ValueError("inputs_embeds must be a torch.Tensor")
        if inputs_embeds.ndim != 3:
            raise ValueError(
                "inputs_embeds must have shape [batch, sequence, hidden]"
            )
        expected_embed_shape = (
            batch_size,
            sequence_length,
            self.hidden_size,
        )
        if tuple(inputs_embeds.shape) != expected_embed_shape:
            raise ValueError(
                "inputs_embeds must have shape "
                f"{expected_embed_shape}, got {tuple(inputs_embeds.shape)}"
            )
        if not torch.is_floating_point(inputs_embeds):
            raise ValueError(
                "inputs_embeds must use a floating-point dtype, got "
                f"{inputs_embeds.dtype}"
            )
        embedding_layer = self.get_input_embeddings()
        try:
            embedding_parameter = next(embedding_layer.parameters())
        except StopIteration as error:
            raise RuntimeError(
                "Thinker input embedding layer has no parameters"
            ) from error
        expected_embedding_dtype = embedding_parameter.dtype
        if inputs_embeds.dtype != expected_embedding_dtype:
            raise ValueError(
                "inputs_embeds dtype must match the Thinker input-embedding "
                f"parameter dtype: expected {expected_embedding_dtype}, got "
                f"{inputs_embeds.dtype}"
            )

        attention_mask = model_inputs["attention_mask"]
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("attention_mask must be a torch.Tensor")
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "attention_mask must have shape "
                f"[{batch_size}, {sequence_length}]"
            )
        self._require_binary_mask_dtype(attention_mask, "attention_mask")

        feature_attention_mask = model_inputs["feature_attention_mask"]
        if not isinstance(feature_attention_mask, torch.Tensor):
            raise ValueError(
                "feature_attention_mask must be a torch.Tensor"
            )
        if feature_attention_mask.ndim != 2:
            raise ValueError(
                "feature_attention_mask must have shape [batch, time]"
            )
        if feature_attention_mask.shape[0] != batch_size:
            raise ValueError(
                "feature_attention_mask batch size must match input_ids"
            )
        if feature_attention_mask.shape[1] == 0:
            raise ValueError(
                "feature_attention_mask must have a non-empty time axis"
            )

        named_tensors = [
            ("inputs_embeds", inputs_embeds),
            ("attention_mask", attention_mask),
            ("feature_attention_mask", feature_attention_mask),
        ]
        if require_labels:
            labels = model_inputs["labels"]
            if not isinstance(labels, torch.Tensor):
                raise ValueError("labels must be a torch.Tensor")
            if labels.ndim != 2 or tuple(labels.shape) != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "labels must have shape "
                    f"[{batch_size}, {sequence_length}]"
                )
            self._require_integer_tensor_dtype(labels, "labels")
            named_tensors.append(("labels", labels))

        for field_name, tensor in named_tensors:
            if tensor.device != input_ids.device:
                raise ValueError(
                    f"{field_name} must be on the same device as input_ids"
                )

        if not torch.isfinite(inputs_embeds).all():
            raise ValueError("inputs_embeds must contain only finite values")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must be binary")
        self.compute_audio_output_lengths(feature_attention_mask)

    def forward_lm(
        self,
        model_inputs: Dict[str, torch.Tensor | None],
    ) -> object:
        with self._state_lock():
            self._validate_lm_model_inputs(
                model_inputs,
                require_labels=True,
            )
            return self.thinker(
                input_ids=model_inputs["input_ids"],
                inputs_embeds=model_inputs["inputs_embeds"],
                attention_mask=model_inputs["attention_mask"],
                feature_attention_mask=model_inputs[
                    "feature_attention_mask"
                ],
                labels=model_inputs["labels"],
                input_features=None,
                use_cache=False,
                return_dict=True,
            )

    @torch.no_grad()
    def generate_from_inputs_embeds(
        self,
        model_inputs: Dict[str, torch.Tensor | None],
        max_new_tokens: int = 32,
        num_beams: int = 1,
    ) -> list[str]:
        with self._state_lock():
            for name, value in (
                ("max_new_tokens", max_new_tokens),
                ("num_beams", num_beams),
            ):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")

            self._validate_lm_model_inputs(
                model_inputs,
                require_labels=False,
            )

            absent = object()
            was_training = self.training
            previous_use_cache = self.thinker.config.use_cache
            previous_rope_deltas = getattr(
                self.thinker,
                "rope_deltas",
                absent,
            )
            try:
                self._set_training_mode_without_freeze(False)
                self.thinker.config.use_cache = True
                self.thinker.rope_deltas = None
                sequences = self.thinker.generate(
                    input_ids=model_inputs["input_ids"],
                    inputs_embeds=model_inputs["inputs_embeds"],
                    attention_mask=model_inputs["attention_mask"],
                    feature_attention_mask=model_inputs[
                        "feature_attention_mask"
                    ],
                    input_features=None,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )

                input_ids = model_inputs["input_ids"]
                prompt_width = input_ids.shape[1]
                batch_size = input_ids.shape[0]
                if (
                    not isinstance(sequences, torch.Tensor)
                    or sequences.ndim != 2
                    or sequences.shape[0] != batch_size
                    or sequences.shape[1] < prompt_width
                ):
                    shape = getattr(sequences, "shape", None)
                    raise ValueError(
                        "generated sequences must be a Tensor with shape "
                        "[input batch, sequence] and sequence width at least "
                        f"the prompt width {prompt_width}, got {shape}"
                    )
                self._require_integer_tensor_dtype(
                    sequences,
                    "generated sequences",
                )

                decoded = self.tokenizer.batch_decode(
                    sequences[:, prompt_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if (
                    not isinstance(decoded, list)
                    or len(decoded) != batch_size
                    or not all(isinstance(item, str) for item in decoded)
                ):
                    raise ValueError(
                        "decoded batch must be a list[str] matching the "
                        f"input batch size {batch_size}"
                    )
                return [item.strip() for item in decoded]
            finally:
                try:
                    self._set_training_mode_without_freeze(was_training)
                finally:
                    try:
                        self.thinker.config.use_cache = previous_use_cache
                    finally:
                        if previous_rope_deltas is absent:
                            if hasattr(self.thinker, "rope_deltas"):
                                delattr(self.thinker, "rope_deltas")
                        else:
                            self.thinker.rope_deltas = previous_rope_deltas

    def build_lm_inputs(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        audio_tokens: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        device: torch.device | str,
        answer_ids_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        left_pad: bool = False,
    ) -> Dict[str, torch.Tensor | None]:
        batch_size = len(prompt_ids_list)
        if batch_size == 0:
            raise ValueError("prompt_ids_list must contain a non-empty batch")
        if not isinstance(audio_tokens, torch.Tensor) or audio_tokens.ndim != 3:
            shape = getattr(audio_tokens, "shape", None)
            raise ValueError(
                "audio_tokens must have shape [batch, time, hidden], "
                f"got {shape}"
            )
        if audio_tokens.shape[0] != batch_size:
            raise ValueError(
                "audio_tokens batch size must match prompt_ids_list batch "
                f"size, got {audio_tokens.shape[0]} and {batch_size}"
            )
        if audio_tokens.shape[2] != self.hidden_size:
            raise ValueError(
                f"audio_tokens hidden dimension must be {self.hidden_size}, "
                f"got {audio_tokens.shape[2]}"
            )
        if not torch.is_floating_point(audio_tokens):
            raise ValueError(
                "audio_tokens must use a floating-point dtype, got "
                f"{audio_tokens.dtype}"
            )
        if not torch.isfinite(audio_tokens).all():
            raise ValueError("audio_tokens must contain only finite values")

        if (
            not isinstance(feature_attention_mask, torch.Tensor)
            or feature_attention_mask.ndim != 2
        ):
            shape = getattr(feature_attention_mask, "shape", None)
            raise ValueError(
                "feature_attention_mask must have shape [batch, time], "
                f"got {shape}"
            )
        if feature_attention_mask.shape[0] != batch_size:
            raise ValueError(
                "feature_attention_mask batch size must match "
                f"prompt_ids_list, got {feature_attention_mask.shape[0]} "
                f"and {batch_size}"
            )
        output_lengths = self.compute_audio_output_lengths(
            feature_attention_mask
        )
        output_lengths_list = output_lengths.to(device="cpu").tolist()
        max_output_length = max(output_lengths_list)
        if audio_tokens.shape[1] != max_output_length:
            raise ValueError(
                "audio_tokens width must equal the maximum computed audio "
                f"output length {max_output_length}, got {audio_tokens.shape[1]}"
            )

        if answer_ids_list is None:
            answers: Sequence[Optional[torch.Tensor]] = [None] * batch_size
        else:
            if len(answer_ids_list) != batch_size:
                raise ValueError(
                    "answer_ids_list batch size must match prompt_ids_list, "
                    f"got {len(answer_ids_list)} and {batch_size}"
                )
            answers = answer_ids_list

        target_device = torch.device(device)
        embedding_layer = self.get_input_embeddings()
        try:
            embedding_parameter = next(embedding_layer.parameters())
        except StopIteration as error:
            raise RuntimeError(
                "Thinker input embedding layer has no parameters"
            ) from error
        embedding_device = embedding_parameter.device
        embedding_dtype = embedding_parameter.dtype

        sequence_ids_for_embedding_list: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        audio_positions_list: list[torch.Tensor] = []
        for batch_index, (prompt_ids, answer_ids) in enumerate(
            zip(prompt_ids_list, answers)
        ):
            if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 1:
                raise ValueError(
                    f"prompt ids at batch {batch_index} must be a 1D tensor"
                )
            if answer_ids is None:
                answer_ids = torch.empty(0, dtype=torch.long)
            elif not isinstance(answer_ids, torch.Tensor) or answer_ids.ndim != 1:
                raise ValueError(
                    f"answer ids at batch {batch_index} must be a 1D tensor"
                )

            prompt_ids_for_embedding = prompt_ids.to(
                device=embedding_device,
                dtype=torch.long,
            )
            answer_ids_for_embedding = answer_ids.to(
                device=embedding_device,
                dtype=torch.long,
            )
            audio_positions = torch.where(
                prompt_ids_for_embedding == self.audio_token_id
            )[0]
            valid_audio_length = output_lengths_list[batch_index]
            if audio_positions.numel() != valid_audio_length:
                raise ValueError(
                    f"batch {batch_index} has {audio_positions.numel()} audio "
                    f"placeholders, expected {valid_audio_length}"
                )
            sequence_ids_for_embedding_list.append(
                torch.cat(
                    (prompt_ids_for_embedding, answer_ids_for_embedding)
                )
            )
            prompt_lengths.append(prompt_ids_for_embedding.numel())
            audio_positions_list.append(audio_positions)

        sequence_lengths = [
            sequence_ids.numel()
            for sequence_ids in sequence_ids_for_embedding_list
        ]
        flat_sequence_ids = torch.cat(sequence_ids_for_embedding_list)
        flat_sequence_embeds = embedding_layer(flat_sequence_ids).to(
            device=target_device,
            dtype=embedding_dtype,
        )
        split_sequence_embeds = flat_sequence_embeds.split(sequence_lengths)

        sequence_ids_list: list[torch.Tensor] = []
        sequence_embeds_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for batch_index, (
            sequence_ids_for_embedding,
            sequence_embeds,
            prompt_length,
            audio_positions,
            valid_audio_length,
        ) in enumerate(
            zip(
                sequence_ids_for_embedding_list,
                split_sequence_embeds,
                prompt_lengths,
                audio_positions_list,
                output_lengths_list,
            )
        ):
            sequence_embeds = sequence_embeds.index_copy(
                0,
                audio_positions.to(device=target_device),
                audio_tokens[batch_index, :valid_audio_length].to(
                    device=target_device,
                    dtype=embedding_dtype,
                ),
            )
            sequence_ids = sequence_ids_for_embedding.to(target_device)
            sequence_ids_list.append(sequence_ids)
            sequence_embeds_list.append(sequence_embeds)
            labels_list.append(
                torch.cat(
                    (
                        torch.full_like(sequence_ids[:prompt_length], -100),
                        sequence_ids[prompt_length:],
                    )
                )
            )

        max_sequence_length = max(ids.shape[0] for ids in sequence_ids_list)
        padded_ids: list[torch.Tensor] = []
        padded_embeds: list[torch.Tensor] = []
        padded_attention: list[torch.Tensor] = []
        padded_labels: list[torch.Tensor] = []
        for sequence_ids, sequence_embeds, labels in zip(
            sequence_ids_list,
            sequence_embeds_list,
            labels_list,
        ):
            padding_length = max_sequence_length - sequence_ids.shape[0]
            ids_padding = torch.full(
                (padding_length,),
                self.pad_token_id,
                dtype=torch.long,
                device=target_device,
            )
            embeds_padding = torch.zeros(
                (padding_length, self.hidden_size),
                dtype=embedding_dtype,
                device=target_device,
            )
            label_padding = torch.full(
                (padding_length,),
                -100,
                dtype=torch.long,
                device=target_device,
            )
            valid_attention = torch.ones(
                sequence_ids.shape[0],
                dtype=torch.long,
                device=target_device,
            )
            padding_attention = torch.zeros(
                padding_length,
                dtype=torch.long,
                device=target_device,
            )
            if left_pad:
                padded_ids.append(torch.cat((ids_padding, sequence_ids)))
                padded_embeds.append(
                    torch.cat((embeds_padding, sequence_embeds), dim=0)
                )
                padded_attention.append(
                    torch.cat((padding_attention, valid_attention))
                )
                padded_labels.append(torch.cat((label_padding, labels)))
            else:
                padded_ids.append(torch.cat((sequence_ids, ids_padding)))
                padded_embeds.append(
                    torch.cat((sequence_embeds, embeds_padding), dim=0)
                )
                padded_attention.append(
                    torch.cat((valid_attention, padding_attention))
                )
                padded_labels.append(torch.cat((labels, label_padding)))

        return {
            "input_ids": torch.stack(padded_ids),
            "inputs_embeds": torch.stack(padded_embeds),
            "attention_mask": torch.stack(padded_attention),
            "feature_attention_mask": feature_attention_mask.to(
                device=target_device,
                dtype=torch.long,
            ),
            "labels": torch.stack(padded_labels),
            "input_features": None,
        }

    @classmethod
    def compute_audio_output_lengths(
        cls,
        feature_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if feature_attention_mask.ndim != 2:
            raise ValueError(
                "feature_attention_mask must have shape [batch, time]"
            )
        if feature_attention_mask.shape[0] == 0:
            raise ValueError(
                "feature_attention_mask must have a non-empty batch axis"
            )
        if feature_attention_mask.shape[1] == 0:
            raise ValueError(
                "feature_attention_mask must have a non-empty time axis"
            )
        cls._require_binary_mask_dtype(
            feature_attention_mask,
            "feature_attention_mask",
        )
        if not torch.all(
            (feature_attention_mask == 0) | (feature_attention_mask == 1)
        ):
            raise ValueError("feature_attention_mask must be binary")

        mask = feature_attention_mask.to(dtype=torch.bool)
        if mask.shape[1] > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError(
                "feature_attention_mask must be a left-aligned prefix"
            )
        input_lengths = mask.sum(-1).to(dtype=torch.long)
        if torch.any(input_lengths == 0):
            raise ValueError(
                "feature_attention_mask requires each audio to contain at "
                "least one valid Mel frame"
            )

        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        return (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2
            + 1
            + (input_lengths // 100) * 13
        )

    def _validate_audio_inputs(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_features.ndim != 3:
            raise ValueError(
                "input_features must have shape [batch, mel, time]"
            )
        batch, mel_bins, width = input_features.shape
        if batch == 0:
            raise ValueError("input_features must have a non-empty batch axis")
        if width == 0:
            raise ValueError("input_features must have a non-empty time axis")
        if mel_bins != self.num_mel_bins:
            raise ValueError(
                f"input_features must have {self.num_mel_bins} Mel bins, "
                f"got {mel_bins}"
            )
        if not torch.isfinite(input_features).all():
            raise ValueError("input_features must contain only finite values")

        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                (batch, width),
                dtype=torch.long,
                device=input_features.device,
            )
        if feature_attention_mask.shape != (batch, width):
            raise ValueError(
                "feature_attention_mask must match input_features "
                "batch/time axes"
            )
        self.compute_audio_output_lengths(feature_attention_mask)
        return input_features, feature_attention_mask.to(dtype=torch.long)

    @torch.no_grad()
    def encode_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        input_features, feature_attention_mask = self._validate_audio_inputs(
            input_features,
            feature_attention_mask,
        )
        output_lengths = self.compute_audio_output_lengths(
            feature_attention_mask
        )

        def stop_before_proj1(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
        ) -> None:
            del module
            if (
                _AUDIO_CAPTURE_INVOCATION.get() is not invocation
                or _AUDIO_CAPTURE_DEPTH.get() != 1
            ):
                return
            raise _PreProjectionCaptured(args[0])

        def enter_audio_encoder(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
        ) -> None:
            del module, args
            if _AUDIO_CAPTURE_INVOCATION.get() is invocation:
                _AUDIO_CAPTURE_DEPTH.set(_AUDIO_CAPTURE_DEPTH.get() + 1)

        def leave_audio_encoder(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
            output: object,
        ) -> None:
            del module, args, output
            if _AUDIO_CAPTURE_INVOCATION.get() is invocation:
                _AUDIO_CAPTURE_DEPTH.set(_AUDIO_CAPTURE_DEPTH.get() - 1)

        audio_parameter = next(self.audio_encoder.parameters())
        invocation = object()
        context_token = _AUDIO_CAPTURE_INVOCATION.set(invocation)
        depth_token = _AUDIO_CAPTURE_DEPTH.set(0)
        handles = []
        try:
            handles.append(
                self.audio_encoder.proj1.register_forward_pre_hook(
                    stop_before_proj1
                )
            )
            handles.append(
                self.audio_encoder.register_forward_pre_hook(
                    enter_audio_encoder,
                    prepend=True,
                )
            )
            handles.append(
                self.audio_encoder.register_forward_hook(
                    leave_audio_encoder,
                    always_call=True,
                )
            )
            self.thinker.get_audio_features(
                input_features.to(
                    device=audio_parameter.device,
                    dtype=audio_parameter.dtype,
                ),
                feature_attention_mask=feature_attention_mask.to(
                    device=audio_parameter.device
                ),
                return_dict=True,
            )
        except _PreProjectionCaptured as captured:
            flat_hidden = captured.hidden_states
        else:
            raise RuntimeError(
                "Audio Tower did not reach proj1 for pre-projection capture"
            )
        finally:
            try:
                for handle in reversed(handles):
                    handle.remove()
            finally:
                try:
                    _AUDIO_CAPTURE_DEPTH.reset(depth_token)
                finally:
                    _AUDIO_CAPTURE_INVOCATION.reset(context_token)

        expected_rows = int(output_lengths.sum().item())
        if flat_hidden.ndim != 2:
            raise RuntimeError(
                "captured Audio Tower hidden states must have shape "
                "[sum(output_lengths), audio_hidden_size], got "
                f"{tuple(flat_hidden.shape)}"
            )
        if flat_hidden.shape[1] != self.audio_hidden_size:
            raise RuntimeError(
                "captured Audio Tower hidden size is "
                f"{flat_hidden.shape[1]}, expected {self.audio_hidden_size}"
            )
        if flat_hidden.shape[0] != expected_rows:
            raise RuntimeError(
                "captured Audio Tower hidden states have "
                f"{flat_hidden.shape[0]} rows, expected {expected_rows} from "
                "output lengths"
            )

        pieces = flat_hidden.split(output_lengths.tolist(), dim=0)
        padded = nn.utils.rnn.pad_sequence(pieces, batch_first=True)
        target_device = device
        if target_device is None:
            target_device = next(
                self.thinker.get_input_embeddings().parameters()
            ).device
        return padded.to(device=target_device, dtype=torch.float32)

    @torch.no_grad()
    def extract_processor_features(
        self,
        waveforms_16k: Sequence[object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(waveforms_16k, Sequence):
            raise ValueError("waveforms_16k must be a sequence of waveforms")
        waveform_count = len(waveforms_16k)
        if waveform_count == 0:
            raise ValueError("waveforms_16k must contain at least one waveform")

        processor_inputs = list(waveforms_16k)
        for waveform_index, waveform in enumerate(processor_inputs):
            try:
                waveform_tensor = torch.as_tensor(waveform)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    f"waveform {waveform_index} must be numeric and tensor-like"
                ) from error
            if waveform_tensor.ndim != 1:
                raise ValueError(
                    f"waveform {waveform_index} must be one-dimensional"
                )
            if waveform_tensor.numel() == 0:
                raise ValueError(
                    f"waveform {waveform_index} must be non-empty"
                )
            try:
                is_finite = torch.isfinite(waveform_tensor).all()
            except RuntimeError as error:
                raise ValueError(
                    f"waveform {waveform_index} must be numeric and tensor-like"
                ) from error
            if not bool(is_finite.item()):
                raise ValueError(
                    f"waveform {waveform_index} must contain only finite values"
                )

        processed = self.processor.feature_extractor(
            processor_inputs,
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if not isinstance(processed, Mapping):
            raise ValueError("processor feature_extractor must return a mapping")
        missing_keys = [
            key
            for key in ("input_features", "attention_mask")
            if key not in processed
        ]
        if missing_keys:
            raise ValueError(
                "processor feature_extractor output is missing: "
                + ", ".join(missing_keys)
            )

        features = processed["input_features"]
        feature_attention_mask = processed["attention_mask"]
        if not isinstance(features, torch.Tensor):
            raise ValueError("input_features must be a torch.Tensor")
        if not isinstance(feature_attention_mask, torch.Tensor):
            raise ValueError("attention_mask must be a torch.Tensor")
        if features.ndim != 3:
            raise ValueError(
                "input_features must have shape [batch, mel, time]"
            )
        batch_size, mel_bins, width = features.shape
        if batch_size != waveform_count:
            raise ValueError(
                "input_features batch size must match the number of waveforms"
            )
        if mel_bins != self.num_mel_bins:
            raise ValueError(
                f"input_features must have {self.num_mel_bins} Mel bins, "
                f"got {mel_bins}"
            )
        if width == 0:
            raise ValueError("input_features must have a non-empty time axis")
        if not torch.is_floating_point(features):
            raise ValueError(
                "input_features must use a floating-point dtype, got "
                f"{features.dtype}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("input_features must contain only finite values")
        if tuple(feature_attention_mask.shape) != (batch_size, width):
            raise ValueError(
                "attention_mask must match input_features batch/time axes"
            )
        self.compute_audio_output_lengths(feature_attention_mask)
        return features, feature_attention_mask.to(dtype=torch.long)

    @staticmethod
    def _validate_comparison_features(
        features: object,
        field_name: str,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise ValueError(f"{field_name} must be a torch.Tensor")
        if features.ndim != 3:
            raise ValueError(
                f"{field_name} must have shape [batch, mel, time]"
            )
        if any(axis == 0 for axis in features.shape):
            raise ValueError(
                f"{field_name} must have non-empty batch, Mel, and time axes"
            )
        if not torch.is_floating_point(features):
            raise ValueError(
                f"{field_name} must use a floating-point dtype, got "
                f"{features.dtype}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"{field_name} must contain only finite values")
        return features

    @torch.no_grad()
    def compare_cached_features(
        self,
        cached_features: torch.Tensor,
        official_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> FeatureEquivalenceReport:
        cached_features = self._validate_comparison_features(
            cached_features,
            "cached_features",
        )
        official_features = self._validate_comparison_features(
            official_features,
            "official_features",
        )
        if not isinstance(feature_attention_mask, torch.Tensor):
            raise ValueError(
                "feature_attention_mask must be a torch.Tensor"
            )
        self.compute_audio_output_lengths(feature_attention_mask)

        mel_shape_equal = bool(
            tuple(cached_features.shape) == tuple(official_features.shape)
        )
        if mel_shape_equal and cached_features.shape[1] != self.num_mel_bins:
            raise ValueError(
                f"cached_features and official_features must have "
                f"{self.num_mel_bins} Mel bins, got "
                f"{cached_features.shape[1]}"
            )

        expected_mask_shape = (
            cached_features.shape[0],
            cached_features.shape[2],
        )
        mask_equal = bool(
            tuple(feature_attention_mask.shape) == expected_mask_shape
            and torch.all(feature_attention_mask == 1).item()
        )

        cached_float = cached_features.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        official_for_cache = (
            official_features.detach()
            .to(dtype=cached_features.dtype)
            .to(device="cpu", dtype=torch.float32)
        )
        quantized_values_finite = bool(
            torch.isfinite(cached_float).all().item()
            and torch.isfinite(official_for_cache).all().item()
        )
        if mel_shape_equal and quantized_values_finite:
            mel_diff = (cached_float - official_for_cache).abs()
            mel_max_abs = float(mel_diff.max().item())
            mel_mean_abs = float(mel_diff.mean().item())
            mel_allclose = bool(
                torch.allclose(
                    cached_float,
                    official_for_cache,
                    rtol=1e-4,
                    atol=1e-5,
                )
            )
        else:
            mel_max_abs = float("inf")
            mel_mean_abs = float("inf")
            mel_allclose = False

        if mel_shape_equal and mask_equal and quantized_values_finite:
            comparison_mask = feature_attention_mask.detach().to(
                device="cpu",
                dtype=torch.long,
            )
            cached_semantic = self.encode_audio_features(
                cached_float,
                comparison_mask,
                device="cpu",
            )
            official_semantic = self.encode_audio_features(
                official_for_cache,
                comparison_mask,
                device="cpu",
            )
            semantic_shape_equal = bool(
                tuple(cached_semantic.shape)
                == tuple(official_semantic.shape)
            )
            semantic_values_comparable = bool(
                semantic_shape_equal
                and cached_semantic.numel() > 0
                and torch.is_floating_point(cached_semantic)
                and torch.is_floating_point(official_semantic)
                and torch.isfinite(cached_semantic).all().item()
                and torch.isfinite(official_semantic).all().item()
            )
            if semantic_values_comparable:
                semantic_diff = (
                    cached_semantic.to(torch.float32)
                    - official_semantic.to(torch.float32)
                ).abs()
                semantic_max_abs = float(semantic_diff.max().item())
                semantic_mean_abs = float(semantic_diff.mean().item())
                semantic_allclose = bool(
                    torch.allclose(
                        cached_semantic.to(torch.float32),
                        official_semantic.to(torch.float32),
                        rtol=1e-2,
                        atol=1e-2,
                    )
                )
            else:
                semantic_max_abs = float("inf")
                semantic_mean_abs = float("inf")
                semantic_allclose = False
        else:
            semantic_shape_equal = False
            semantic_max_abs = float("inf")
            semantic_mean_abs = float("inf")
            semantic_allclose = False

        passed = bool(
            mel_shape_equal
            and mask_equal
            and mel_allclose
            and semantic_shape_equal
            and semantic_allclose
        )
        return FeatureEquivalenceReport(
            mel_shape_equal=mel_shape_equal,
            mask_equal=mask_equal,
            mel_max_abs=mel_max_abs,
            mel_mean_abs=mel_mean_abs,
            mel_allclose=mel_allclose,
            semantic_shape_equal=semantic_shape_equal,
            semantic_max_abs=semantic_max_abs,
            semantic_mean_abs=semantic_mean_abs,
            semantic_allclose=semantic_allclose,
            passed=passed,
        )

    @torch.no_grad()
    def require_cached_feature_equivalence(
        self,
        cached_features: torch.Tensor,
        official_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> FeatureEquivalenceReport:
        report = self.compare_cached_features(
            cached_features,
            official_features,
            feature_attention_mask,
        )
        if not report.passed:
            metrics = ", ".join(
                f"{field_name}={getattr(report, field_name)}"
                for field_name in report.__dataclass_fields__
            )
            raise ValueError(
                "cached semantic features are not equivalent to the Qwen3 "
                f"processor: {metrics}"
            )
        return report

    def trainable_parameter_names(self) -> List[str]:
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def freeze(self) -> None:
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        if self.enable_llm_lora:
            self.audit_lora_targets()
            for _, module in self._named_modules_including_aliases(
                self.thinker.model
            ):
                if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                    continue
                adapter_name = self.LORA_ADAPTER_NAME
                for parameter in module.lora_A[adapter_name].parameters():
                    parameter.requires_grad = True
                for parameter in module.lora_B[adapter_name].parameters():
                    parameter.requires_grad = True
        forbidden = [
            name
            for name in self.trainable_parameter_names()
            if "audio_tower" in name or ".visual." in name or "lm_head" in name
        ]
        if forbidden:
            raise RuntimeError(
                "frozen modules contain trainable parameters: " + ", ".join(forbidden)
            )

    def train(self, mode: bool = True) -> "FrozenQwen3OmniGenerativeBackbone":
        with self._state_lock():
            super().train(mode)
            self.freeze()
            self.audio_encoder.eval()
            self.vision_encoder.eval()
            self.thinker.lm_head.eval()
            self.thinker.model.train(mode and self.enable_llm_lora)
            return self
