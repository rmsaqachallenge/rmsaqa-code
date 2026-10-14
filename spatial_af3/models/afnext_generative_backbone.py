from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration
from transformers.models.musicflamingo.modeling_musicflamingo import (
    apply_rotary_time_emb,
)

from ..layered_mcq import build_mcq_prompt_body


def _prepare_musicflamingo_inputs_for_generation(
    model: MusicFlamingoForConditionalGeneration,
    input_ids: torch.LongTensor,
    next_sequence_length: int | None = None,
    past_key_values=None,
    attention_mask: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    is_first_iteration: bool | None = False,
    **kwargs,
):
    """Expose ``inputs_embeds`` while preserving MusicFlamingo's official prep."""
    input_features = kwargs.pop("input_features", None)
    input_features_mask = kwargs.pop("input_features_mask", None)
    model_inputs = super(
        MusicFlamingoForConditionalGeneration,
        model,
    ).prepare_inputs_for_generation(
        input_ids,
        next_sequence_length=next_sequence_length,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        is_first_iteration=is_first_iteration,
        **kwargs,
    )
    if is_first_iteration or not model_inputs.get("use_cache", False):
        if input_features is not None:
            model_inputs["input_features"] = input_features
        if input_features_mask is not None:
            model_inputs["input_features_mask"] = input_features_mask
    return model_inputs


class FrozenAFNextGenerativeBackbone(nn.Module):
    """AF3-Next length helpers operating on two-dimensional binary masks."""

    REQUIRED_FILES = (
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")

    def __init__(
        self,
        afnext_root: str | Path,
        text_dtype: torch.dtype = torch.bfloat16,
        audio_dtype: torch.dtype = torch.bfloat16,
        enable_llm_lora: bool = True,
        llm_lora_r: int = 16,
        llm_lora_alpha: int = 16,
        llm_lora_dropout: float = 0.05,
        llm_lora_num_layers: int = 28,
    ) -> None:
        super().__init__()
        self.afnext_root = self._validate_checkpoint_root(afnext_root)
        self.text_dtype = text_dtype
        self.audio_dtype = audio_dtype
        self.enable_llm_lora = bool(enable_llm_lora)
        self.llm_lora_r = int(llm_lora_r)
        self.llm_lora_alpha = int(llm_lora_alpha)
        self.llm_lora_dropout = float(llm_lora_dropout)
        self.llm_lora_num_layers = int(llm_lora_num_layers)
        if self.enable_llm_lora:
            self._validate_lora_config()
        self.llm_has_trainable_lora = False

        self.processor = AutoProcessor.from_pretrained(
            self.afnext_root,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.llm = MusicFlamingoForConditionalGeneration.from_pretrained(
            self.afnext_root,
            local_files_only=True,
            torch_dtype=text_dtype,
            low_cpu_mem_usage=True,
        )
        self.llm.prepare_inputs_for_generation = MethodType(
            _prepare_musicflamingo_inputs_for_generation,
            self.llm,
        )

        self.audio_encoder.to(dtype=audio_dtype)
        self.sound_projector.to(dtype=text_dtype)

        text_config = self.llm.config.text_config
        audio_config = self.llm.config.audio_config
        self.hidden_size = int(text_config.hidden_size)
        self.audio_hidden_size = int(audio_config.hidden_size)
        self.num_mel_bins = int(audio_config.num_mel_bins)
        self.max_mel_frames = 2 * int(audio_config.max_source_positions)
        self.pad_token_id = self._validated_token_id(
            self.tokenizer.pad_token_id,
            "pad_token_id",
        )
        self.eos_token_id = self._validated_token_id(
            self.tokenizer.eos_token_id,
            "eos_token_id",
        )
        self.audio_token_id = self._validated_token_id(
            self.llm.config.audio_token_id,
            "config.audio_token_id for <sound>",
        )
        self.audio_bos_token_id = self._validated_token_id(
            self.llm.config.audio_bos_token_id,
            "config.audio_bos_token_id",
        )
        self.audio_eos_token_id = self._validated_token_id(
            self.llm.config.audio_eos_token_id,
            "config.audio_eos_token_id",
        )
        tokenizer_vocab = self.tokenizer.get_vocab()
        if "<sound>" not in tokenizer_vocab:
            raise ValueError("Tokenizer does not contain <sound> special token.")
        sound_token_id = self._validated_token_id(
            tokenizer_vocab["<sound>"],
            "vocabulary <sound> token id",
        )
        converted_sound_token_id = self._validated_token_id(
            self.tokenizer.convert_tokens_to_ids("<sound>"),
            "converted <sound> token id",
        )
        if not (
            converted_sound_token_id == sound_token_id == self.audio_token_id
        ):
            raise ValueError(
                "Tokenizer vocabulary, token conversion, and config.audio_token_id "
                "must agree for <sound>."
            )
        self.sound_token_id = sound_token_id
        self.audio_bos_token = self._validated_audio_boundary_token(
            getattr(self.processor, "audio_bos_token", None),
            getattr(self.processor, "audio_bos_token_id", None),
            self.audio_bos_token_id,
            "audio_bos_token",
            tokenizer_vocab,
        )
        self.audio_eos_token = self._validated_audio_boundary_token(
            getattr(self.processor, "audio_eos_token", None),
            getattr(self.processor, "audio_eos_token_id", None),
            self.audio_eos_token_id,
            "audio_eos_token",
            tokenizer_vocab,
        )

        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        if self.enable_llm_lora:
            self._inject_trainable_llm_lora()

        self.llm.config.use_cache = False
        self.llm.config.text_config.use_cache = False
        self.llm.gradient_checkpointing_enable()
        self.freeze()
        self.train(False)

    @classmethod
    def _validate_checkpoint_root(cls, afnext_root: str | Path) -> Path:
        root = Path(afnext_root)
        if not root.is_dir():
            raise FileNotFoundError(f"AF3-Next checkpoint directory does not exist: {root}")
        missing = [
            filename for filename in cls.REQUIRED_FILES if not (root / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"AF3-Next checkpoint is missing required files: {', '.join(missing)}"
            )
        if not any((root / filename).is_file() for filename in cls.WEIGHT_FILES):
            choices = " or ".join(cls.WEIGHT_FILES)
            raise FileNotFoundError(f"AF3-Next checkpoint requires {choices}: {root}")
        return root

    def _validate_lora_config(self) -> None:
        if self.llm_lora_r <= 0:
            raise ValueError("llm_lora_r must be positive.")
        if self.llm_lora_alpha <= 0:
            raise ValueError("llm_lora_alpha must be positive.")
        if self.llm_lora_num_layers <= 0:
            raise ValueError("llm_lora_num_layers must be positive.")
        if not 0.0 <= self.llm_lora_dropout < 1.0:
            raise ValueError("llm_lora_dropout must be in the range [0, 1).")

    @staticmethod
    def _validated_token_id(value: object, label: str) -> int:
        if value is None:
            raise ValueError(f"{label} must not be None.")
        try:
            token_id = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} must be an integer token id.") from error
        if token_id < 0:
            raise ValueError(f"{label} must be non-negative, got {token_id}.")
        return token_id

    def _validated_audio_boundary_token(
        self,
        token: object,
        processor_token_id: object,
        expected_token_id: int,
        label: str,
        tokenizer_vocab: Dict[str, int],
    ) -> str:
        if not isinstance(token, str) or not token:
            raise ValueError(f"processor.{label} must be a non-empty token string.")
        if token not in tokenizer_vocab:
            raise ValueError(f"processor.{label} {token!r} is missing from tokenizer vocabulary.")
        vocab_token_id = self._validated_token_id(
            tokenizer_vocab[token],
            f"tokenizer vocabulary id for processor.{label}",
        )
        converted_token_id = self._validated_token_id(
            self.tokenizer.convert_tokens_to_ids(token),
            f"converted token id for processor.{label}",
        )
        validated_processor_token_id = self._validated_token_id(
            processor_token_id,
            f"processor.{label}_id",
        )
        if not (
            validated_processor_token_id
            == vocab_token_id
            == converted_token_id
            == expected_token_id
        ):
            raise ValueError(
                f"processor.{label}/{label}_id, tokenizer vocabulary, token conversion, and "
                f"config id must agree at token id {expected_token_id}."
            )
        return token

    def _inject_trainable_llm_lora(self) -> None:
        num_layers = int(self.llm.config.text_config.num_hidden_layers)
        target_layer_count = min(self.llm_lora_num_layers, num_layers)
        layer_alternation = "|".join(str(index) for index in range(target_layer_count))
        target_modules = (
            rf"layers[.](?:{layer_alternation})[.]"
            r"self_attn[.](?:q_proj|k_proj|v_proj|o_proj)"
        )
        peft_config = LoraConfig(
            r=self.llm_lora_r,
            lora_alpha=self.llm_lora_alpha,
            lora_dropout=self.llm_lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules,
        )
        self.llm.model.language_model = get_peft_model(
            self.llm.model.language_model,
            peft_config,
        )
        self.llm_has_trainable_lora = True

    @staticmethod
    def _is_lora_parameter(name: str) -> bool:
        return "lora_" in name

    @property
    def audio_encoder(self) -> nn.Module:
        return self.llm.model.audio_tower

    @property
    def sound_projector(self) -> nn.Module:
        return self.llm.model.multi_modal_projector

    def freeze(self) -> None:
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        if self.llm_has_trainable_lora:
            for name, parameter in self.llm.model.language_model.named_parameters():
                if self._is_lora_parameter(name):
                    parameter.requires_grad = True

        trainable = self.trainable_parameter_names()
        if self.enable_llm_lora and not trainable:
            raise RuntimeError(
                "LLM LoRA is enabled but no trainable LoRA parameters were found."
            )
        forbidden = [
            name
            for name in trainable
            if "audio_tower" in name or "multi_modal_projector" in name
        ]
        if forbidden:
            raise RuntimeError(
                "Frozen semantic modules contain trainable parameters: " + ", ".join(forbidden)
            )

    def train(self, mode: bool = True) -> "FrozenAFNextGenerativeBackbone":
        super().train(mode)
        self.freeze()
        self.audio_encoder.eval()
        self.sound_projector.eval()
        self.llm.model.pos_emb.eval()
        self.llm.model.language_model.train(mode and self.llm_has_trainable_lora)
        return self

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def build_user_prompt(
        self,
        question: str,
        options: Optional[Dict[str, str]] = None,
    ) -> str:
        question = question.strip()
        header = (
            f"{self.audio_bos_token}<sound>{self.audio_eos_token}\n"
            "You are given a first-order ambisonics audio clip. Use the audio "
            "content faithfully, and use spatial cues only when the question "
            "requires them.\n"
        )
        if options is not None:
            return header + build_mcq_prompt_body(question=question, options=options)
        return header + f"Question: {question}\nAnswer briefly and directly."

    def build_chat_prompt_ids_from_messages(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> torch.Tensor:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

    def build_chat_prompt_ids(self, user_prompt: str) -> torch.Tensor:
        return self.build_chat_prompt_ids_from_messages(
            [{"role": "user", "content": user_prompt}]
        )

    def tokenize_answer_ids(self, answer_text: str) -> torch.Tensor:
        eos_token = self.tokenizer.eos_token
        if not eos_token:
            raise ValueError("Tokenizer eos_token must be defined to tokenize answers.")
        return self.tokenizer(
            answer_text.strip() + eos_token,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

    @torch.no_grad()
    def embed_token_ids(
        self,
        token_ids: torch.Tensor,
        device: torch.device | str,
    ) -> torch.Tensor:
        if token_ids.numel() == 0:
            return torch.zeros(
                (0, self.hidden_size),
                dtype=self.text_dtype,
                device=device,
            )
        return self.get_input_embeddings()(token_ids.to(device=device)).to(
            dtype=self.text_dtype
        )

    def build_inputs_embeds(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        audio_tokens: torch.Tensor,
        spatial_tokens: Optional[torch.Tensor],
        device: torch.device | str,
        answer_ids_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        only_first_answer_token_loss: bool = False,
        audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(prompt_ids_list)
        if batch_size == 0:
            raise ValueError("prompt_ids_list must contain a non-empty batch.")
        if audio_tokens.ndim != 3:
            raise ValueError(
                "audio_tokens must have shape [batch, time, hidden], "
                f"got {tuple(audio_tokens.shape)}"
            )
        if audio_tokens.shape[0] != batch_size:
            raise ValueError(
                "audio_tokens batch size must match prompt_ids_list batch size, "
                f"got {audio_tokens.shape[0]} and {batch_size}"
            )
        if audio_tokens.shape[2] != self.hidden_size:
            raise ValueError(
                f"audio_tokens hidden dimension must be {self.hidden_size}, "
                f"got {audio_tokens.shape[2]}"
            )
        if audio_tokens.shape[1] == 0:
            raise ValueError("audio_tokens must have a non-empty time dimension.")
        audio_attention_mask = self._validated_embedding_attention_mask(
            audio_attention_mask,
            audio_tokens,
            "audio_attention_mask",
        )
        if spatial_tokens is not None:
            if spatial_tokens.ndim != 3:
                raise ValueError(
                    "spatial_tokens must have shape [batch, time, hidden], "
                    f"got {tuple(spatial_tokens.shape)}"
                )
            if spatial_tokens.shape[0] != batch_size:
                raise ValueError(
                    "spatial_tokens batch size must match prompt_ids_list batch size, "
                    f"got {spatial_tokens.shape[0]} and {batch_size}"
                )
            if spatial_tokens.shape[2] != self.hidden_size:
                raise ValueError(
                    f"spatial_tokens hidden dimension must be {self.hidden_size}, "
                    f"got {spatial_tokens.shape[2]}"
                )
            spatial_attention_mask = self._validated_embedding_attention_mask(
                spatial_attention_mask,
                spatial_tokens,
                "spatial_attention_mask",
            )
        elif spatial_attention_mask is not None:
            raise ValueError(
                "spatial_attention_mask cannot be provided without spatial_tokens."
            )
        if answer_ids_list is not None and len(answer_ids_list) != batch_size:
            raise ValueError(
                "answer_ids_list length must match the prompt batch size, "
                f"got {len(answer_ids_list)} and {batch_size}"
            )

        sequence_embeds: List[torch.Tensor] = []
        sequence_labels: List[torch.Tensor] = []
        for index, prompt_ids in enumerate(prompt_ids_list):
            if prompt_ids.ndim != 1:
                raise ValueError(
                    "Each prompt_ids tensor must be one-dimensional, "
                    f"got shape {tuple(prompt_ids.shape)} at batch index {index}"
                )
            prompt_ids = prompt_ids.to(device=device)
            sound_positions = torch.nonzero(
                prompt_ids == self.sound_token_id,
                as_tuple=False,
            ).flatten()
            if sound_positions.numel() != 1:
                raise ValueError(
                    "Each prompt must contain exactly one <sound> token; "
                    f"batch item {index} contains {sound_positions.numel()}."
                )
            sound_position = int(sound_positions.item())
            pre_embed = self.embed_token_ids(prompt_ids[:sound_position], device)
            post_embed = self.embed_token_ids(prompt_ids[sound_position + 1 :], device)
            multimodal_embed = audio_tokens[index].to(
                device=device,
                dtype=self.text_dtype,
            )
            multimodal_embed = multimodal_embed[
                audio_attention_mask[index].to(device=device)
            ]
            if spatial_tokens is not None:
                valid_spatial_tokens = spatial_tokens[index].to(
                    device=device,
                    dtype=self.text_dtype,
                )[spatial_attention_mask[index].to(device=device)]
                multimodal_embed = torch.cat(
                    [
                        multimodal_embed,
                        valid_spatial_tokens,
                    ],
                    dim=0,
                )

            parts = [pre_embed, multimodal_embed, post_embed]
            labels = [
                torch.full(
                    (part.shape[0],),
                    -100,
                    dtype=torch.long,
                    device=device,
                )
                for part in parts
            ]
            if answer_ids_list is not None and answer_ids_list[index] is not None:
                answer_ids = answer_ids_list[index]
                if answer_ids.ndim != 1:
                    raise ValueError(
                        "Each answer_ids tensor must be one-dimensional, "
                        f"got shape {tuple(answer_ids.shape)} at batch index {index}"
                    )
                if answer_ids.numel() == 0:
                    raise ValueError(
                        f"Supervised answer at batch index {index} must not be empty."
                    )
                answer_ids = answer_ids.to(device=device, dtype=torch.long)
                parts.append(self.embed_token_ids(answer_ids, device))
                if only_first_answer_token_loss:
                    answer_labels = torch.full_like(answer_ids, -100)
                    if answer_ids.numel() > 0:
                        answer_labels[0] = answer_ids[0]
                else:
                    answer_labels = answer_ids
                labels.append(answer_labels)

            sequence_embeds.append(torch.cat(parts, dim=0))
            sequence_labels.append(torch.cat(labels, dim=0))

        max_length = max(item.shape[0] for item in sequence_embeds)
        padded_embeds = torch.zeros(
            (batch_size, max_length, self.hidden_size),
            dtype=self.text_dtype,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=device,
        )
        padded_labels = torch.full(
            (batch_size, max_length),
            -100,
            dtype=torch.long,
            device=device,
        )
        for index, (embeds, labels) in enumerate(
            zip(sequence_embeds, sequence_labels)
        ):
            length = embeds.shape[0]
            padded_embeds[index, :length] = embeds
            attention_mask[index, :length] = 1
            padded_labels[index, :length] = labels

        return {
            "inputs_embeds": padded_embeds,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

    @staticmethod
    def _validated_embedding_attention_mask(
        mask: Optional[torch.Tensor],
        embeddings: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        expected_shape = embeddings.shape[:2]
        if mask is None:
            return torch.ones(
                expected_shape,
                dtype=torch.bool,
                device=embeddings.device,
            )
        if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"{label} must have shape {tuple(expected_shape)}, got {tuple(mask.shape)}"
            )
        if mask.is_complex() or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{label} must be binary (0/1).")
        return mask.to(dtype=torch.bool)

    @torch.no_grad()
    def generate_from_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 12,
        temperature: float = 0.0,
        num_beams: int = 1,
    ) -> List[str]:
        """Generate and decode HF's generated-only IDs for ``inputs_embeds``."""
        was_training = self.training
        config_use_cache = self.llm.config.use_cache
        text_config_use_cache = self.llm.config.text_config.use_cache
        try:
            self.eval()
            self.llm.config.use_cache = True
            self.llm.config.text_config.use_cache = True
            inputs_embeds, attention_mask = self._left_pad_generation_inputs(
                inputs_embeds,
                attention_mask,
            )
            do_sample = temperature > 0
            generation_arguments = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
            if do_sample:
                generation_arguments.update(
                    temperature=temperature,
                    top_p=1.0,
                    top_k=0,
                )
            outputs = self.llm.generate(**generation_arguments)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        finally:
            self.llm.config.use_cache = config_use_cache
            self.llm.config.text_config.use_cache = text_config_use_cache
            self.train(was_training)

    @staticmethod
    def _left_pad_generation_inputs(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds.ndim != 3:
            raise ValueError(
                "inputs_embeds must have shape [batch, time, hidden], "
                f"got {tuple(inputs_embeds.shape)}"
            )
        if attention_mask.ndim != 2 or attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError(
                "attention_mask must match inputs_embeds batch/time dimensions, "
                f"got {tuple(attention_mask.shape)} and {tuple(inputs_embeds.shape)}"
            )
        if attention_mask.is_complex() or not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError("attention_mask must be binary (0/1).")
        valid_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        valid_lengths = valid_mask.sum(dim=1)
        if torch.any(valid_lengths == 0):
            raise ValueError("Each generation item must contain at least one valid embedding.")
        target_length = int(valid_lengths.max().item())
        left_padded_embeds = torch.zeros(
            (inputs_embeds.shape[0], target_length, inputs_embeds.shape[2]),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        left_padded_mask = torch.zeros(
            (inputs_embeds.shape[0], target_length),
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        for index in range(inputs_embeds.shape[0]):
            valid_embeds = inputs_embeds[index][valid_mask[index]]
            length = valid_embeds.shape[0]
            left_padded_embeds[index, -length:] = valid_embeds
            left_padded_mask[index, -length:] = 1
        return left_padded_embeds, left_padded_mask

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def _prepare_audio_features(
        self,
        input_features: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_features.ndim != 3:
            raise ValueError(
                "input_features must be three-dimensional [batch, mel bins, time], "
                f"got shape {tuple(input_features.shape)}"
            )

        batch_size, mel_bins, width = input_features.shape
        if batch_size == 0:
            raise ValueError(
                f"input_features must not have an empty batch, got shape {tuple(input_features.shape)}"
            )
        if mel_bins != self.num_mel_bins:
            raise ValueError(
                f"input_features must have {self.num_mel_bins} mel bins, got {mel_bins}"
            )
        if width == 0:
            raise ValueError(
                f"input_features must have a non-empty time axis, got shape {tuple(input_features.shape)}"
            )
        if width > self.max_mel_frames:
            raise ValueError(
                f"input_features width {width} exceeds max_mel_frames {self.max_mel_frames}"
            )

        if mask is None:
            mask = torch.ones(
                (batch_size, width),
                dtype=torch.bool,
                device=input_features.device,
            )
        elif mask.ndim != 2 or tuple(mask.shape) != (batch_size, width):
            raise ValueError(
                "mask must have shape [batch, time] matching input_features, "
                f"got {tuple(mask.shape)} for input shape {tuple(input_features.shape)}"
            )

        if mask.is_complex() or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"mask must be binary (0/1), got shape {tuple(mask.shape)}")
        binary_mask = mask.to(dtype=torch.bool)
        if width > 1 and torch.any(binary_mask[:, 1:] & ~binary_mask[:, :-1]):
            raise ValueError("mask must be a left-aligned contiguous prefix of valid frames")

        output_lengths = self.compute_audio_output_lengths(binary_mask)
        pad_width = self.max_mel_frames - width
        feature_mask = binary_mask.to(device=input_features.device).unsqueeze(1)
        masked_features = input_features.masked_fill(~feature_mask, 0)
        padded_features = F.pad(masked_features, (0, pad_width))
        padded_mask = F.pad(binary_mask, (0, pad_width), value=False)

        encoder_parameter = next(self.audio_encoder.parameters())
        encoder_device = encoder_parameter.device
        encoder_dtype = encoder_parameter.dtype
        return (
            padded_features.to(device=encoder_device, dtype=encoder_dtype),
            padded_mask.to(device=encoder_device),
            output_lengths.to(device=encoder_device),
        )

    def _apply_official_rote(
        self,
        hidden_states: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_valid_length = int(output_lengths.max().item())
        if hidden_states.shape[0] != output_lengths.shape[0]:
            raise ValueError(
                "audio encoder batch size does not match output lengths: "
                f"{hidden_states.shape[0]} versus {output_lengths.shape[0]}"
            )
        if hidden_states.shape[1] < max_valid_length:
            raise ValueError(
                "audio encoder output is shorter than the required valid length: "
                f"{hidden_states.shape[1]} versus {max_valid_length}"
            )

        positions = torch.arange(max_valid_length, device=output_lengths.device)
        valid_mask = positions.unsqueeze(0) < output_lengths.unsqueeze(1)
        placeholder_ids = torch.full(
            valid_mask.shape,
            self.pad_token_id,
            dtype=torch.long,
            device=output_lengths.device,
        )
        placeholder_ids.masked_fill_(valid_mask, self.sound_token_id)

        timestamps = self.llm.model._build_audio_timestamps(
            placeholder_ids,
            output_lengths,
            max_valid_length,
        )
        cropped_hidden_states = hidden_states[:, :max_valid_length]
        cos, sin = self.llm.model.pos_emb(
            timestamps.to(device=cropped_hidden_states.device),
            seq_len=max_valid_length,
        )
        rotated_hidden_states = apply_rotary_time_emb(
            cropped_hidden_states,
            cos,
            sin,
        )
        return rotated_hidden_states.masked_fill(
            ~valid_mask.to(device=rotated_hidden_states.device).unsqueeze(-1),
            0,
        )

    @torch.no_grad()
    def encode_audio_features(
        self,
        input_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        padded_features, padded_mask, output_lengths = self._prepare_audio_features(
            input_features,
            mask,
        )
        hidden_states = self.audio_encoder(
            input_features=padded_features,
            input_features_mask=padded_mask,
            return_dict=True,
        ).last_hidden_state
        hidden_states = self._apply_official_rote(hidden_states, output_lengths)

        target_device = (
            device
            if device is not None
            else next(self.get_input_embeddings().parameters()).device
        )
        return hidden_states.to(device=target_device, dtype=torch.float32)

    @torch.no_grad()
    def encode_audio_tokens(
        self,
        input_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        projector_device = next(self.sound_projector.parameters()).device
        hidden_states = self.encode_audio_features(
            input_features,
            mask=mask,
            device=projector_device,
        ).to(device=projector_device, dtype=self.text_dtype)
        audio_tokens = self.sound_projector(hidden_states)
        if mask is None:
            mask = torch.ones(
                (input_features.shape[0], input_features.shape[2]),
                dtype=torch.bool,
                device=input_features.device,
            )
        output_mask = self.build_audio_attention_mask(
            mask,
            output_width=audio_tokens.shape[1],
        ).to(device=audio_tokens.device)
        audio_tokens = audio_tokens.masked_fill(~output_mask.unsqueeze(-1), 0)
        target_device = (
            device
            if device is not None
            else next(self.get_input_embeddings().parameters()).device
        )
        return audio_tokens.to(device=target_device, dtype=self.text_dtype)

    @staticmethod
    def compute_audio_output_lengths(mask: torch.Tensor) -> torch.Tensor:
        """Return official AF3-Next lengths as ``torch.long`` on ``mask.device``.

        ``mask`` must be a non-empty batch of boolean or numeric 0/1 values.
        """
        if mask.ndim != 2:
            raise ValueError(f"mask must be two-dimensional, got shape {tuple(mask.shape)}")
        if mask.shape[0] == 0:
            raise ValueError(f"mask must not have an empty batch, got shape {tuple(mask.shape)}")
        if mask.is_complex():
            raise ValueError(
                "mask must use a boolean, integer, or floating dtype, "
                f"got {mask.dtype} for shape {tuple(mask.shape)}"
            )
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(
                f"mask must be binary (0/1), got shape {tuple(mask.shape)}"
            )

        valid_frames = mask.to(dtype=torch.long).sum(dim=1)
        if not torch.all(valid_frames > 0):
            raise ValueError(
                "mask must contain at least one valid frame per item, "
                f"got valid lengths {valid_frames.tolist()} for shape {tuple(mask.shape)}"
            )

        conv_lengths = (valid_frames - 1) // 2 + 1
        output_lengths = (conv_lengths - 2) // 2 + 1
        if not torch.all(output_lengths > 0):
            raise ValueError(
                "AFNext convolution and pooling require positive output lengths, "
                f"got {output_lengths.tolist()} from mask shape {tuple(mask.shape)}"
            )
        return output_lengths

    @staticmethod
    def build_audio_attention_mask(
        mask: torch.Tensor, output_width: int | None = None
    ) -> torch.Tensor:
        """Build a boolean prefix mask on the binary input mask's device.

        ``output_width`` defaults to the maximum output length; an explicit
        value may add right padding but cannot be smaller than that maximum.
        """
        output_lengths = FrozenAFNextGenerativeBackbone.compute_audio_output_lengths(mask)
        required_width = int(output_lengths.max().item())
        if output_width is None:
            output_width = required_width
        elif output_width < required_width:
            raise ValueError(
                f"output_width {output_width} is smaller than required width {required_width}"
            )

        positions = torch.arange(output_width, device=output_lengths.device)
        return positions.unsqueeze(0) < output_lengths.unsqueeze(1)
