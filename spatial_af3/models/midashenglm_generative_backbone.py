"""Frozen MiDashengLM loader with raw/native audio encoding and state audit.

The module owns local checkpoint loading, model component aliases, frozen/LoRA
parameter auditing, and the raw Dasheng (``encode_audio_features``) and native
projected (``encode_audio_tokens``) audio interfaces. Prompt, embedding
injection, and language-model task APIs are implemented in later phases.
"""

from __future__ import annotations

import json
import os
from operator import index
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torchaudio.functional as torchaudio_functional
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from ..layered_mcq import build_mcq_prompt_body


class FrozenMiDashengLMGenerativeBackbone(nn.Module):
    """Load MiDashengLM from a local checkpoint with a strict frozen contract."""

    REQUIRED_FILES = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    )
    LORA_ADAPTER_NAME = "default"
    _FORBIDDEN_TRAINABLE_TOKENS = (
        "audio_encoder",
        "audio_projector",
        "embed_tokens",
        "lm_head",
        ".norm",
        "norm.",
    )

    def __init__(
        self,
        midashenglm_root: str | Path,
        *,
        text_dtype: torch.dtype = torch.bfloat16,
        audio_dtype: torch.dtype = torch.float32,
        enable_llm_lora: bool = True,
        llm_lora_r: int = 16,
        llm_lora_alpha: int = 16,
        llm_lora_dropout: float = 0.05,
        llm_lora_num_layers: int | None = None,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        super().__init__()
        self.midashenglm_root = self._validate_checkpoint_root(midashenglm_root)
        self.text_dtype = text_dtype
        self.audio_dtype = audio_dtype
        self.enable_llm_lora = bool(enable_llm_lora)
        self.llm_lora_r = int(llm_lora_r)
        self.llm_lora_alpha = int(llm_lora_alpha)
        self.llm_lora_dropout = float(llm_lora_dropout)
        self.llm_lora_num_layers = (
            None if llm_lora_num_layers is None else int(llm_lora_num_layers)
        )
        # MiDashengLM's remote Dasheng frontend constructs its mel filterbank
        # during ``from_pretrained``.  Transformers' meta-device init
        # (``low_cpu_mem_usage=True``) makes that construction mix CPU and
        # meta tensors inside torchaudio.melscale_fbanks, causing a hard
        # device-mismatch error before any weights are loaded.  Keep the
        # safe eager path as the default; callers with a patched remote-code
        # implementation can opt back into low-memory loading explicitly.
        self.low_cpu_mem_usage = bool(low_cpu_mem_usage)
        self._freeze_initialized = False
        self.lora_target_audit: dict[int, set[str]] = {}

        config = AutoConfig.from_pretrained(
            self.midashenglm_root,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.midashenglm_root,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        # Construct the model on a real CPU device.  Some Accelerate/
        # Transformers combinations leave a process-wide ``meta`` default
        # device active even when ``low_cpu_mem_usage=False``; the custom
        # Dasheng frontend creates torchaudio filterbanks during ``__init__``
        # and cannot safely mix those meta tensors with CPU tensors.
        with torch.device("cpu"):
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.midashenglm_root,
                config=config,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=text_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
            )

        try:
            self.llm.audio_encoder
            self.llm.audio_projector
            self.llm.decoder
        except AttributeError as exc:
            raise AttributeError(
                "MiDashengLM checkpoint must expose audio_encoder, "
                "audio_projector, and decoder modules"
            ) from exc
        # The checkpoint's custom Dasheng frontend owns non-persistent DSP
        # buffers.  Older remote-code versions can construct them inside
        # Transformers' meta initialization context, leaving NaNs that are
        # not represented in the safetensors shards.  Rebuild invalid buffers
        # on a real CPU device before the model is moved to its execution
        # device.
        self._repair_audio_frontend_buffers(self.llm.audio_encoder)
        self.audio_encoder.to(dtype=audio_dtype)

        model_config = getattr(self.llm, "config", config)
        text_config = self._config_attr(model_config, "text_config", config)
        audio_config = self._config_attr(
            model_config,
            "audio_encoder_config",
            config,
        )
        self.hidden_size = self._positive_int(
            self._first_config_value(text_config, "hidden_size"),
            "text hidden size",
        )
        self.audio_hidden_size = self._positive_int(
            self._first_config_value(
                audio_config,
                "embed_dim",
                fallback=getattr(self.audio_encoder, "embed_dim", None),
            ),
            "audio hidden size",
        )
        self.num_mel_bins = self._positive_int(
            self._first_config_value(
                audio_config,
                "n_mels",
                "num_mel_bins",
            ),
            "number of mel bins",
        )
        self.audio_token_id = self._nonnegative_int(
            self._first_config_value(model_config, "audio_token_id"),
            "audio token id",
        )
        self.audio_token = self._required_token_string(
            "audio_token", processor=self.processor, tokenizer=self.tokenizer
        )
        self.audio_bos_token = self._required_token_string(
            "audio_bos_token", processor=self.processor, tokenizer=self.tokenizer
        )
        self.audio_eos_token = self._required_token_string(
            "audio_eos_token", processor=self.processor, tokenizer=self.tokenizer
        )
        converter = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if not callable(converter):
            raise ValueError("tokenizer must define convert_tokens_to_ids")
        converted_audio_id = self._nonnegative_int(
            converter(self.audio_token), "tokenizer audio token id"
        )
        if converted_audio_id != self.audio_token_id:
            raise ValueError(
                "tokenizer audio token id does not match config audio_token_id: "
                f"{converted_audio_id} != {self.audio_token_id}"
            )
        self.subsample_factor = self._positive_int(
            self._first_config_value(model_config, "subsample_factor"),
            "audio subsample factor",
        )
        self.sample_rate = self._positive_int(
            self._first_config_value(audio_config, "sample_rate"),
            "audio sample rate",
        )
        self.pad_token_id = self._tokenizer_id(
            "pad_token_id",
            fallback=self._first_config_value(text_config, "pad_token_id"),
        )
        self.eos_token_id = self._tokenizer_id(
            "eos_token_id",
            fallback=self._first_config_value(text_config, "eos_token_id"),
        )

        self._freeze_base_parameters()
        if self.enable_llm_lora:
            self._validate_lora_config()
            self._inject_trainable_llm_lora()
        self.freeze()
        self.train(False)

    @classmethod
    def _validate_checkpoint_root(cls, value: str | Path) -> Path:
        root = Path(value)
        if not root.is_dir():
            raise FileNotFoundError(
                f"MiDashengLM checkpoint directory does not exist: {root}"
            )
        missing = [name for name in cls.REQUIRED_FILES if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "MiDashengLM checkpoint is missing required files: "
                + ", ".join(missing)
            )
        index_path = root / "model.safetensors.index.json"
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid safetensors index JSON: {index_path}") from exc
        weight_map = index_data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in safetensors index: {index_path}")
        shard_names = sorted({str(name) for name in weight_map.values()})
        missing_shards = [name for name in shard_names if not (root / name).is_file()]
        if missing_shards:
            raise FileNotFoundError(
                "MiDashengLM checkpoint is missing indexed weight shards: "
                + ", ".join(missing_shards)
            )
        return root

    @staticmethod
    def _config_attr(config: Any, name: str, fallback: Any = None) -> Any:
        value = getattr(config, name, None)
        if value is None and isinstance(config, dict):
            value = config.get(name)
        return fallback if value is None else value

    @classmethod
    def _first_config_value(
        cls,
        config: Any,
        *names: str,
        fallback: Any = None,
    ) -> Any:
        for name in names:
            value = cls._config_attr(config, name)
            if value is not None:
                return value
        return fallback

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a positive integer") from exc
        if result <= 0:
            raise ValueError(f"{label} must be a positive integer")
        return result

    @staticmethod
    def _nonnegative_int(value: Any, label: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a non-negative integer") from exc
        if result < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return result

    def _tokenizer_id(self, name: str, *, fallback: Any) -> int:
        value = getattr(self.tokenizer, name, None)
        if value is None:
            value = fallback
        return self._nonnegative_int(value, f"tokenizer {name}")

    @staticmethod
    def _required_token_string(name: str, *, processor: Any, tokenizer: Any) -> str:
        value = getattr(processor, name, None)
        if value is None:
            value = getattr(tokenizer, name, None)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value

    def _freeze_base_parameters(self) -> None:
        for parameter in self.llm.parameters():
            parameter.requires_grad = False

    def _validate_lora_config(self) -> None:
        if self.llm_lora_r <= 0 or self.llm_lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.llm_lora_num_layers is not None and self.llm_lora_num_layers <= 0:
            raise ValueError("llm_lora_num_layers must be positive when provided")
        if not 0.0 <= self.llm_lora_dropout < 1.0:
            raise ValueError("llm_lora_dropout must be in [0, 1)")

    def _decoder_layer_count(self) -> int:
        for container in (
            getattr(getattr(self.decoder, "model", None), "layers", None),
            getattr(self.decoder, "layers", None),
        ):
            if container is not None:
                try:
                    return len(container)
                except TypeError:
                    pass
        layer_indexes = set()
        for name, _ in self.decoder.named_modules():
            parts = name.split(".")
            if "layers" in parts:
                index = parts[parts.index("layers") + 1]
                if index.isdigit():
                    layer_indexes.add(int(index))
        if not layer_indexes:
            raise ValueError("decoder does not expose a layers module")
        return max(layer_indexes) + 1

    def _inject_trainable_llm_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "PEFT is required when enable_llm_lora=True; install the `peft` "
                "package or disable LoRA"
            ) from exc
        actual_layers = self._decoder_layer_count()
        requested = actual_layers if self.llm_lora_num_layers is None else self.llm_lora_num_layers
        if requested > actual_layers:
            raise ValueError(
                f"requested {requested} LoRA layers, but decoder has {actual_layers}"
            )
        config = LoraConfig(
            r=self.llm_lora_r,
            lora_alpha=self.llm_lora_alpha,
            lora_dropout=self.llm_lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            layers_to_transform=list(range(requested)),
            layers_pattern="layers",
        )
        self.llm.decoder = get_peft_model(self.llm.decoder, config)
        self.lora_target_audit = self._audit_lora_targets(requested)

    def _audit_lora_targets(self, requested_layers: int) -> dict[int, set[str]]:
        import re

        pattern = re.compile(
            r"(?:^|\.)layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
        )
        expected = {
            (layer, projection)
            for layer in range(requested_layers)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        found: set[tuple[int, str]] = set()
        unexpected: set[str] = set()
        for name, module in self.llm.decoder.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            match = pattern.search(name)
            if match is None:
                unexpected.add(name)
                continue
            found.add((int(match.group(1)), match.group(2)))
        if found != expected or unexpected:
            missing = sorted(expected - found)
            extras = sorted((found - expected)) + sorted(unexpected)
            raise RuntimeError(
                "LoRA target audit failed: "
                f"missing={missing}; unexpected={extras}"
            )
        return {
            layer: {projection for index, projection in found if index == layer}
            for layer in range(requested_layers)
        }

    @staticmethod
    def _is_lora_parameter(name: str) -> bool:
        return ".lora_A." in name or ".lora_B." in name

    def freeze(self) -> None:
        if self._freeze_initialized:
            forbidden = [
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
                and any(token in name for token in self._FORBIDDEN_TRAINABLE_TOKENS)
            ]
            if forbidden:
                raise RuntimeError(
                    "frozen modules contain trainable parameters: "
                    + ", ".join(forbidden)
                )

        self._freeze_base_parameters()
        if self.enable_llm_lora:
            for name, parameter in self.llm.named_parameters():
                if self._is_lora_parameter(name):
                    parameter.requires_grad = True

        forbidden = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and any(token in name for token in self._FORBIDDEN_TRAINABLE_TOKENS)
        ]
        if forbidden:
            raise RuntimeError(
                "frozen modules contain trainable parameters: "
                + ", ".join(forbidden)
            )
        self._freeze_initialized = True

    def trainable_parameter_names(self) -> list[str]:
        return [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]

    def train(self, mode: bool = True) -> "FrozenMiDashengLMGenerativeBackbone":
        super().train(mode)
        self.freeze()
        self.audio_encoder.eval()
        self.audio_projector.eval()
        lm_head = getattr(self.decoder, "lm_head", None)
        if lm_head is not None:
            lm_head.eval()
        return self

    @property
    def audio_encoder(self) -> nn.Module:
        return self.llm.audio_encoder

    @property
    def audio_projector(self) -> nn.Module:
        return self.llm.audio_projector

    @property
    def decoder(self) -> nn.Module:
        return self.llm.decoder

    def get_input_embeddings(self) -> nn.Module:
        """Return the decoder token embedding table (bare or PEFT wrapped)."""
        decoder = self.decoder
        getter = getattr(decoder, "get_input_embeddings", None)
        if callable(getter):
            embeddings = getter()
            if embeddings is not None:
                return embeddings
        model = getattr(decoder, "model", None)
        embeddings = getattr(model, "embed_tokens", None)
        if embeddings is None:
            embeddings = getattr(decoder, "embed_tokens", None)
        if embeddings is None:
            raise AttributeError("decoder does not expose input token embeddings")
        return embeddings

    def build_user_prompt(
        self,
        question: str,
        options: Optional[Dict[str, str]] = None,
    ) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty after stripping")
        question = question.strip()
        header = (
            "You are given a first-order ambisonics audio clip. Use the audio "
            "content faithfully, and use spatial cues only when the question "
            "requires them.\n"
        )
        if options is not None:
            if not isinstance(options, dict) or not options:
                raise ValueError("options must be a non-empty dict")
            normalized: Dict[str, str] = {}
            for raw_key, raw_text in options.items():
                key = str(raw_key).strip().upper()
                text = str(raw_text).strip()
                if not key:
                    raise ValueError("each option key must be non-empty after stripping")
                if key in normalized:
                    raise ValueError(f"duplicate option key after normalization: {key}")
                if not text:
                    raise ValueError(f"option text for {key} must be non-empty after stripping")
                normalized[key] = text
            return header + build_mcq_prompt_body(question=question, options=normalized)
        return header + f"Question: {question}\nAnswer briefly and directly."

    def build_chat_prompt_ids(
        self,
        question: str,
        audio_length: int,
        add_generation_prompt: bool = True,
    ) -> torch.Tensor:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty after stripping")
        return self.build_chat_prompt_ids_from_messages(
            [{"role": "user", "content": question.strip()}],
            audio_length=audio_length,
            add_generation_prompt=add_generation_prompt,
        )

    def build_chat_prompt_ids_from_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        audio_length: int,
        add_generation_prompt: bool = True,
    ) -> torch.Tensor:
        if isinstance(audio_length, bool):
            raise ValueError("audio_length must be a positive integer")
        try:
            audio_length = index(audio_length)
        except TypeError as exc:
            raise ValueError("audio_length must be a positive integer") from exc
        if audio_length <= 0:
            raise ValueError("audio_length must be a positive integer")
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a non-empty sequence")
        normalized_messages = [dict(message) for message in messages]
        first_user = next(
            (i for i, message in enumerate(normalized_messages) if message.get("role") == "user"),
            None,
        )
        if first_user is None:
            raise ValueError("messages must contain a user turn")
        audio_token = self.audio_token
        first_content = normalized_messages[first_user].get("content")
        if isinstance(first_content, str):
            first_content = [
                {"type": "audio", "audio": "cached://foa-w-channel"},
                {"type": "text", "text": first_content.strip()},
            ]
        elif isinstance(first_content, (list, tuple)):
            first_content = [dict(part) if isinstance(part, dict) else part for part in first_content]
            if not any(
                isinstance(part, dict) and part.get("type") == "audio"
                for part in first_content
            ):
                first_content.insert(0, {"type": "audio", "audio": "cached://foa-w-channel"})
        else:
            raise ValueError("first user message content must be text or content parts")
        normalized_messages[first_user]["content"] = first_content
        rendered = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered, str):
            raise ValueError("processor.apply_chat_template must return a string")
        if rendered.count(audio_token) != 1:
            raise ValueError(
                "rendered chat template must contain exactly one unexpanded audio token"
            )
        rendered = rendered.replace(audio_token, audio_token * audio_length, 1)
        output = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )
        token_ids = getattr(output, "input_ids", None)
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("tokenizer input_ids must have non-empty shape [1, length]")
        if token_ids.shape[1] == 0:
            raise ValueError("tokenizer input_ids must have non-empty shape [1, length]")
        try:
            torch.iinfo(token_ids.dtype)
        except TypeError as exc:
            raise ValueError("tokenizer input_ids must use a PyTorch integer dtype") from exc
        prompt_ids = token_ids[0].to(dtype=torch.long)
        actual = int((prompt_ids == self.audio_token_id).sum().item())
        if actual != audio_length:
            raise ValueError(
                f"tokenizer produced {actual} audio token ids, expected {audio_length} audio token ids"
            )
        positions = torch.where(prompt_ids == self.audio_token_id)[0]
        expected = torch.arange(positions[0], positions[0] + audio_length, device=positions.device)
        if not torch.equal(positions, expected):
            raise ValueError("tokenizer-produced audio token ids must form one adjacent span")
        converter = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if not callable(converter):
            raise ValueError("tokenizer must define convert_tokens_to_ids")
        bos_id = self._nonnegative_int(converter(self.audio_bos_token), "audio BOS token id")
        eos_id = self._nonnegative_int(converter(self.audio_eos_token), "audio EOS token id")
        if positions[0].item() == 0 or positions[-1].item() + 1 >= prompt_ids.numel():
            raise ValueError("audio token span must be surrounded by audio BOS/EOS tokens")
        if prompt_ids[positions[0] - 1].item() != bos_id:
            raise ValueError("audio token span is not preceded by audio BOS token")
        if prompt_ids[positions[-1] + 1].item() != eos_id:
            raise ValueError("audio token span is not followed by audio EOS token")
        return prompt_ids

    def build_inputs_embeds(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        audio_tokens: torch.Tensor,
        answer_ids_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        device: torch.device | str = "cpu",
    ) -> Dict[str, torch.Tensor | None]:
        batch_size = len(prompt_ids_list)
        if batch_size == 0:
            raise ValueError("prompt_ids_list must contain a non-empty batch")
        if not isinstance(audio_tokens, torch.Tensor) or audio_tokens.ndim != 3:
            raise ValueError("audio_tokens must have shape [batch, time, hidden]")
        if audio_tokens.shape[0] != batch_size:
            raise ValueError("audio_tokens batch size must match prompt_ids_list batch size")
        if audio_tokens.shape[1] == 0 or audio_tokens.shape[2] != self.hidden_size:
            raise ValueError("audio_tokens must have non-empty time and hidden dimensions")
        if not audio_tokens.is_floating_point() or audio_tokens.is_complex() or not torch.isfinite(audio_tokens).all().item():
            raise ValueError("audio_tokens must contain finite floating-point values")
        if answer_ids_list is not None and len(answer_ids_list) != batch_size:
            raise ValueError("answer_ids_list length must match prompt_ids_list batch size")
        emb_layer = self.get_input_embeddings()
        sequences = []
        labels_list = []
        input_ids_sequences = []
        for i, prompt in enumerate(prompt_ids_list):
            if not isinstance(prompt, torch.Tensor) or prompt.ndim != 1 or prompt.numel() == 0:
                raise ValueError(f"prompt_ids at batch index {i} must be a non-empty 1D tensor")
            try:
                torch.iinfo(prompt.dtype)
            except TypeError as exc:
                raise ValueError("prompt_ids must use an integer dtype") from exc
            prompt = prompt.to(device=device, dtype=torch.long)
            positions = torch.where(prompt == self.audio_token_id)[0]
            if positions.numel() != audio_tokens.shape[1]:
                raise ValueError(f"prompt at batch index {i} must contain exactly {audio_tokens.shape[1]} audio tokens")
            expected = torch.arange(positions[0], positions[0] + audio_tokens.shape[1], device=positions.device)
            if not torch.equal(positions, expected):
                raise ValueError("audio token ids must form one adjacent span")
            answer = None if answer_ids_list is None else answer_ids_list[i]
            if answer is not None:
                if not isinstance(answer, torch.Tensor) or answer.ndim != 1 or answer.numel() == 0:
                    raise ValueError(f"answer_ids at batch index {i} must be a non-empty 1D tensor")
                try:
                    torch.iinfo(answer.dtype)
                except TypeError as exc:
                    raise ValueError("answer_ids must use an integer dtype") from exc
                answer = answer.to(device=device, dtype=torch.long)
                ids = torch.cat([prompt, answer])
            else:
                ids = prompt
            embeds = emb_layer(ids).to(dtype=self.text_dtype).clone()
            if embeds.shape[-1] != self.hidden_size:
                raise ValueError("decoder embedding dimension does not match configured hidden size")
            start = int(positions[0].item())
            embeds[start : start + audio_tokens.shape[1]] = audio_tokens[i].to(device=device, dtype=embeds.dtype)
            labels = torch.full((ids.numel(),), -100, dtype=torch.long, device=device)
            if answer is not None:
                labels[-answer.numel() :] = answer
            sequences.append(embeds)
            input_ids_sequences.append(ids)
            labels_list.append(labels)
        max_len = max(item.numel() for item in input_ids_sequences)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long, device=device)
        padded_embeds = torch.zeros((batch_size, max_len, self.hidden_size), dtype=self.text_dtype, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
        for i, (ids, embeds, labels) in enumerate(zip(input_ids_sequences, sequences, labels_list)):
            n = ids.numel()
            input_ids[i, -n:] = ids
            padded_embeds[i, -n:] = embeds
            attention_mask[i, -n:] = 1
            padded_labels[i, -n:] = labels
        return {"input_ids": input_ids, "inputs_embeds": padded_embeds, "attention_mask": attention_mask, "labels": padded_labels, "input_values": None, "audio_length": None}

    def forward_lm(self, model_inputs: Dict[str, Any]) -> Any:
        """Run the frozen decoder on prepared multimodal embeddings."""
        if not isinstance(model_inputs, dict):
            raise TypeError("model_inputs must be a dictionary")
        for key in ("inputs_embeds", "attention_mask", "labels"):
            if key not in model_inputs or not isinstance(model_inputs[key], torch.Tensor):
                raise ValueError(f"model_inputs must contain tensor key {key!r}")
        kwargs = {
            "input_ids": None,
            "inputs_embeds": model_inputs["inputs_embeds"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": model_inputs["labels"],
            "input_values": model_inputs.get("input_values"),
            "use_cache": False,
            "return_dict": True,
        }
        try:
            output = self.decoder(**kwargs)
        except TypeError as exc:
            # Standard Qwen2 decoders do not expose MiDasheng's input_values kwarg.
            if "input_values" not in str(exc):
                raise
            kwargs.pop("input_values")
            output = self.decoder(**kwargs)
        loss = getattr(output, "loss", None)
        logits = getattr(output, "logits", None)
        if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss).all().item():
            raise ValueError("decoder output must contain a finite loss tensor")
        if not isinstance(logits, torch.Tensor):
            raise ValueError("decoder output must contain logits tensor")
        return output

    @torch.no_grad()
    def generate_from_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 12,
        num_beams: int = 1,
    ) -> list[str]:
        if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [batch, time, hidden]")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("attention_mask must match inputs_embeds batch/time dimensions")
        try:
            max_new_tokens = index(max_new_tokens)
            num_beams = index(num_beams)
        except TypeError as exc:
            raise ValueError("max_new_tokens and num_beams must be positive integers") from exc
        if max_new_tokens <= 0 or num_beams <= 0:
            raise ValueError("max_new_tokens and num_beams must be positive")
        if attention_mask.ndim != 2 or attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("attention_mask must match inputs_embeds batch/time dimensions")
        if attention_mask.dtype != torch.bool:
            try:
                torch.iinfo(attention_mask.dtype)
            except TypeError as exc:
                raise ValueError("attention_mask must use a bool or integer dtype") from exc
        decoder = self.decoder
        config = getattr(decoder, "config", None)
        previous_training = self.training
        had_cache = config is not None and hasattr(config, "use_cache")
        previous_cache = getattr(config, "use_cache", None)
        had_rope = hasattr(decoder, "rope_deltas")
        previous_rope = getattr(decoder, "rope_deltas", None)
        prompt_width = inputs_embeds.shape[1]
        try:
            self.eval()
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = True
            if had_rope:
                decoder.rope_deltas = None
            generated = decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
                raise ValueError("decoder.generate must return token IDs with shape [batch, time]")
            if generated.shape[0] != inputs_embeds.shape[0]:
                print(
                    "[MiDashengLM generation debug] "
                    f"rank={os.environ.get('RANK', '?')} "
                    f"inputs_embeds={tuple(inputs_embeds.shape)} "
                    f"attention_mask={tuple(attention_mask.shape)} "
                    f"generated={tuple(generated.shape)} "
                    f"max_new_tokens={max_new_tokens} "
                    f"num_beams={num_beams}",
                    flush=True,
                )
                raise ValueError(
                    "decoder.generate output batch is incompatible with prompt: "
                    f"expected batch={inputs_embeds.shape[0]}; "
                    f"got shape={tuple(generated.shape)}"
                )
            # Some decoder implementations return the complete prompt plus
            # continuation, while others return continuation tokens only when
            # called with ``inputs_embeds``.  Decode the latter as-is.
            if generated.shape[1] >= prompt_width:
                generated = generated[:, prompt_width:]
            decoded = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded, (list, tuple)) or len(decoded) != inputs_embeds.shape[0]:
                raise ValueError("tokenizer.batch_decode must return one string per batch item")
            return list(decoded)
        finally:
            if had_cache:
                config.use_cache = previous_cache
            elif config is not None and hasattr(config, "use_cache"):
                delattr(config, "use_cache")
            if had_rope:
                decoder.rope_deltas = previous_rope
            elif hasattr(decoder, "rope_deltas"):
                delattr(decoder, "rope_deltas")
            self.train(previous_training)

    @staticmethod
    def _validate_audio_inputs(
        input_values: torch.Tensor,
        audio_length: torch.Tensor | list[int] | tuple[int, ...] | None,
    ) -> torch.Tensor | None:
        """Validate waveform and optional sample lengths before model execution."""
        if not isinstance(input_values, torch.Tensor):
            raise TypeError("input_values must be a torch.Tensor")
        if input_values.ndim != 2:
            raise ValueError(
                "input_values must have shape [B, samples], "
                f"got {tuple(input_values.shape)}"
            )
        if input_values.shape[0] == 0 or input_values.shape[1] == 0:
            raise ValueError("input_values must have a non-empty batch and samples")
        if not input_values.is_floating_point() or input_values.is_complex():
            raise ValueError(
                "input_values must use a finite floating-point dtype, "
                f"got {input_values.dtype}"
            )
        if not torch.isfinite(input_values).all().item():
            raise ValueError("input_values must contain only finite values")
        if audio_length is None:
            return None
        if isinstance(audio_length, torch.Tensor):
            lengths = audio_length
        else:
            lengths = torch.as_tensor(audio_length)
        if lengths.ndim != 1 or lengths.shape[0] != input_values.shape[0]:
            raise ValueError(
                "audio_length must have shape [B] matching input_values, "
                f"got {tuple(lengths.shape)} for batch {input_values.shape[0]}"
            )
        integer_dtypes = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
        if lengths.dtype == torch.bool or lengths.dtype not in integer_dtypes:
            raise ValueError(
                "audio_length must use a positive integer dtype, "
                f"got {lengths.dtype}"
            )
        lengths = lengths.to(dtype=torch.long)
        if not torch.all(lengths > 0).item():
            raise ValueError("audio_length must contain positive values")
        if not torch.all(lengths <= input_values.shape[1]).item():
            raise ValueError(
                "audio_length values must not exceed input_values sample width "
                f"({input_values.shape[1]})"
            )
        return lengths

    @staticmethod
    def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return fallback

    @staticmethod
    def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return fallback

    @staticmethod
    def _repair_audio_frontend_buffers(audio_encoder: nn.Module) -> None:
        """Ensure Dasheng's non-persistent frontend buffers are materialized.

        These buffers are DSP constants rather than learned parameters, so
        they may be absent from a checkpoint and can be accidentally created
        as uninitialized/meta tensors by older Transformers loading paths.
        Recreate them on CPU when missing, meta-backed, malformed, or
        non-finite; the caller can then move the encoder normally.
        """
        frontend = getattr(audio_encoder, "front_end", None)
        config = getattr(frontend, "config", None)
        if frontend is None or config is None:
            return

        def invalid(value: object, expected_shape: tuple[int, ...]) -> bool:
            if not isinstance(value, torch.Tensor):
                return True
            if tuple(value.shape) != expected_shape or value.device.type == "meta":
                return True
            try:
                return not torch.isfinite(value).all().item()
            except (RuntimeError, ValueError):
                return True

        win_length = int(config.win_length)
        n_freqs = int(config.n_fft) // 2 + 1
        n_mels = int(config.n_mels)
        window_shape = (win_length,)
        mel_shape = (n_freqs, n_mels)
        window = getattr(frontend, "spectrogram_window", None)
        mel_fbanks = getattr(frontend, "melscale_fbanks", None)

        if invalid(window, window_shape):
            frontend.spectrogram_window = torch.hann_window(
                win_length,
                device="cpu",
                dtype=torch.float32,
            )
        if invalid(mel_fbanks, mel_shape):
            frontend.melscale_fbanks = torchaudio_functional.melscale_fbanks(
                n_freqs=n_freqs,
                f_min=float(config.f_min),
                f_max=float(config.f_max),
                n_mels=n_mels,
                sample_rate=int(config.sample_rate),
            ).to(device="cpu", dtype=torch.float32)

    def _encode_audio_bundle(
        self,
        input_values: torch.Tensor,
        audio_length: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        *,
        include_native: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        """Encode raw Dasheng features and, optionally, native projected tokens."""
        lengths = self._validate_audio_inputs(input_values, audio_length)
        encoder_device = self._module_device(self.audio_encoder, input_values.device)
        encoder_values = input_values.to(device=encoder_device, dtype=self.audio_dtype)
        encoder_lengths = None if lengths is None else lengths.to(device=encoder_device)
        with torch.no_grad():
            encoded = self.audio_encoder(encoder_values, encoder_lengths)
        if isinstance(encoded, (tuple, list)):
            if not encoded:
                raise ValueError("audio_encoder returned an empty output")
            raw_features = encoded[0]
            raw_mask = encoded[1] if len(encoded) > 1 else None
        else:
            raw_features = getattr(encoded, "last_hidden_state", encoded)
            raw_mask = getattr(encoded, "attention_mask", None)
        if not isinstance(raw_features, torch.Tensor) or raw_features.ndim != 3:
            raise ValueError(
                "audio_encoder must return features with shape [B, T_enc, 1280]"
            )
        if raw_features.shape[0] != input_values.shape[0]:
            raise ValueError("audio_encoder output batch does not match input_values")
        if raw_features.shape[-1] != self.audio_hidden_size:
            raise ValueError(
                "audio_encoder output width does not match configured audio hidden size: "
                f"expected {self.audio_hidden_size}, got {raw_features.shape[-1]}"
            )
        if raw_mask is not None:
            if not isinstance(raw_mask, torch.Tensor) or raw_mask.shape != raw_features.shape[:2]:
                raise ValueError(
                    "audio_encoder mask must have shape [B, T_enc] matching features"
                )
        native_tokens = native_mask = None
        if include_native:
            projector_device = self._module_device(self.audio_projector, raw_features.device)
            projector_dtype = self._module_dtype(self.audio_projector, self.audio_dtype)
            projector_features = raw_features.to(
                device=projector_device, dtype=projector_dtype
            )
            # Forward the encoder mask exactly as the official MiDashengLM model
            # does so variable-length groups retain their ``any`` semantics.
            projector_mask = None if raw_mask is None else raw_mask.to(device=projector_device)
            with torch.no_grad():
                projected = self.audio_projector(projector_features, projector_mask)
            if isinstance(projected, (tuple, list)):
                if not projected:
                    raise ValueError("audio_projector returned an empty output")
                native_tokens = projected[0]
                native_mask = projected[1] if len(projected) > 1 else None
            else:
                native_tokens = projected
            if not isinstance(native_tokens, torch.Tensor) or native_tokens.ndim != 3:
                raise ValueError(
                    "audio_projector must return tokens with shape [B, T_mi, hidden]"
                )
            if native_tokens.shape[0] != input_values.shape[0]:
                raise ValueError("audio_projector output batch does not match input_values")
            if native_tokens.shape[-1] != self.hidden_size:
                raise ValueError(
                    "audio_projector output width does not match configured hidden size: "
                    f"expected {self.hidden_size}, got {native_tokens.shape[-1]}"
                )
            expected_native_length = raw_features.shape[1] // self.subsample_factor
            if native_tokens.shape[1] != expected_native_length:
                raise RuntimeError(
                    "audio_projector output length is inconsistent with the configured "
                    f"subsample factor: expected {expected_native_length}, "
                    f"got {native_tokens.shape[1]}"
                )
            if native_mask is not None:
                if not isinstance(native_mask, torch.Tensor) or native_mask.shape != native_tokens.shape[:2]:
                    raise ValueError(
                        "audio_projector mask must have shape [B, T_mi] matching tokens"
                    )
            native_tokens = native_tokens.to(dtype=torch.float32)
            if native_mask is not None:
                native_mask = native_mask.to(dtype=torch.long)
        return {
            "raw_features": raw_features.to(dtype=torch.float32),
            "raw_mask": None if raw_mask is None else raw_mask,
            "native_tokens": native_tokens,
            "native_mask": native_mask,
        }

    @torch.no_grad()
    def encode_audio_features(
        self,
        input_values: torch.Tensor,
        audio_length: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Return raw Dasheng encoder features before the official projector."""
        bundle = self._encode_audio_bundle(
            input_values, audio_length, include_native=False
        )
        return bundle["raw_features"]  # type: ignore[return-value]

    @torch.no_grad()
    def encode_audio_tokens(
        self,
        input_values: torch.Tensor,
        audio_length: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Return native MiDashengLM audio tokens from the official projector."""
        bundle = self._encode_audio_bundle(
            input_values, audio_length, include_native=True
        )
        return bundle["native_tokens"]  # type: ignore[return-value]
