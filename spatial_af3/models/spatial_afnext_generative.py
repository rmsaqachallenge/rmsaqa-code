from __future__ import annotations

import math
from numbers import Real
from typing import Dict, List, Optional, Sequence, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layered_mcq import build_mcq_prompt_body, format_mcq_target
from .afnext_generative_backbone import FrozenAFNextGenerativeBackbone
from .spatial_encoder import ResnetConformerTokenEncoder


class FreeformGenerationOutput(TypedDict):
    layer1_text: List[str]
    layer2_text: List[str]
    layer1_gate: List[float]
    layer2_gate: List[float]


class AFNextTemporalConcatFusion(nn.Module):
    def __init__(
        self,
        semantic_dim: int = 1280,
        spatial_dim: int = 1024,
        llm_dim: int = 3584,
    ) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.spatial_dim = int(spatial_dim)
        self.llm_dim = int(llm_dim)
        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        self.concat_project = nn.Linear(
            self.semantic_dim + self.spatial_dim,
            self.llm_dim,
        )
        self._init_concat_project()

    def _init_concat_project(self) -> None:
        with torch.no_grad():
            self.concat_project.weight.zero_()
            self.concat_project.bias.zero_()
            nn.init.kaiming_uniform_(
                self.concat_project.weight[:, : self.semantic_dim],
                a=math.sqrt(5),
            )

    def forward(
        self,
        semantic_audio: torch.Tensor,
        spatial_seq: torch.Tensor,
        semantic_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if semantic_audio.ndim != 3 or spatial_seq.ndim != 3:
            raise ValueError("semantic_audio and spatial_seq must be three-dimensional")
        if semantic_audio.shape[0] != spatial_seq.shape[0]:
            raise ValueError("semantic_audio and spatial_seq batch size must match")
        if semantic_audio.shape[1] == 0 or spatial_seq.shape[1] == 0:
            raise ValueError(
                "semantic_audio and spatial_seq time dimensions must be non-empty"
            )
        if semantic_audio.shape[2] != self.semantic_dim:
            raise ValueError(
                f"Expected semantic dimension {self.semantic_dim}, "
                f"got {semantic_audio.shape[2]}"
            )
        if spatial_seq.shape[2] != self.spatial_dim:
            raise ValueError(
                f"Expected Spatial dimension {self.spatial_dim}, "
                f"got {spatial_seq.shape[2]}"
            )

        if semantic_attention_mask is None:
            semantic_attention_mask = torch.ones(
                semantic_audio.shape[:2],
                dtype=torch.bool,
                device=semantic_audio.device,
            )
        elif tuple(semantic_attention_mask.shape) != tuple(semantic_audio.shape[:2]):
            raise ValueError(
                "semantic_attention_mask must have shape [batch, semantic_time]"
            )
        elif semantic_attention_mask.dtype != torch.bool and not torch.all(
            (semantic_attention_mask == 0) | (semantic_attention_mask == 1)
        ):
            raise ValueError("semantic_attention_mask must be binary")

        aligned_spatial = F.adaptive_avg_pool1d(
            spatial_seq.transpose(1, 2),
            output_size=semantic_audio.shape[1],
        ).transpose(1, 2)
        normalized_spatial = self.spatial_norm(aligned_spatial)
        fused_audio = self.concat_project(
            torch.cat([semantic_audio, normalized_spatial], dim=-1)
        )
        gate = torch.ones(
            semantic_audio.shape[:2],
            dtype=semantic_audio.dtype,
            device=semantic_audio.device,
        )
        return {
            "fused_audio": fused_audio,
            "aligned_spatial": aligned_spatial,
            "normalized_spatial": normalized_spatial,
            "gate": gate,
            "audio_content_mask": semantic_attention_mask.to(
                device=semantic_audio.device,
                dtype=torch.bool,
            ),
        }


class SpatialAFNextGenerativeModel(nn.Module):
    def __init__(
        self,
        afnext_root: str,
        text_dtype: torch.dtype = torch.bfloat16,
        audio_dtype: torch.dtype = torch.bfloat16,
        enable_llm_lora: bool = True,
        llm_lora_r: int = 16,
        llm_lora_alpha: int = 16,
        llm_lora_dropout: float = 0.05,
        llm_lora_num_layers: int = 28,
        spatial_encoder_dim: int = 1024,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.backbone = FrozenAFNextGenerativeBackbone(
            afnext_root=afnext_root,
            text_dtype=text_dtype,
            audio_dtype=audio_dtype,
            enable_llm_lora=enable_llm_lora,
            llm_lora_r=llm_lora_r,
            llm_lora_alpha=llm_lora_alpha,
            llm_lora_dropout=llm_lora_dropout,
            llm_lora_num_layers=llm_lora_num_layers,
        )
        self.spatial_encoder = ResnetConformerTokenEncoder(
            encoder_dim=spatial_encoder_dim,
        )
        self.fusion = AFNextTemporalConcatFusion(
            semantic_dim=self.backbone.audio_hidden_size,
            spatial_dim=spatial_encoder_dim,
            llm_dim=self.backbone.hidden_size,
        )
        self.loss_weights = {
            "lm_layer1": 1.0,
            "lm_layer2": 1.0,
        }
        if loss_weights is not None:
            unknown_keys = sorted(set(loss_weights) - set(self.loss_weights))
            if unknown_keys:
                raise ValueError(
                    "Unknown loss weight key(s): "
                    f"{', '.join(unknown_keys)}. Expected lm_layer1 and/or lm_layer2."
                )
            for name, weight in loss_weights.items():
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, Real)
                    or not math.isfinite(weight)
                    or weight < 0
                ):
                    raise ValueError(
                        f"Loss weight {name!r} must be a finite non-negative real number."
                    )
                self.loss_weights[name] = float(weight)
        if not any(weight > 0 for weight in self.loss_weights.values()):
            raise ValueError("At least one loss weight must be greater than zero.")

    def train(self, mode: bool = True) -> "SpatialAFNextGenerativeModel":
        super().train(mode)
        self.backbone.train(mode)
        return self

    @staticmethod
    def _normalize_history_text(text: str) -> str:
        text = (text or "").strip()
        return " ".join(text.split())

    def _build_followup_user_prompt(
        self,
        question: str,
        options: Dict[str, str],
    ) -> str:
        return (
            "Based on the same audio clip and the previous answer, answer the follow-up question.\n"
            f"{build_mcq_prompt_body(question=question, options=options)}"
        )

    @staticmethod
    def _format_mcq_targets(
        correct_options: Sequence[str],
        options_batch: Sequence[Dict[str, str]],
        answer_texts: Sequence[str],
    ) -> List[str]:
        batch_lengths = (
            len(correct_options),
            len(options_batch),
            len(answer_texts),
        )
        if len(set(batch_lengths)) != 1:
            raise ValueError(
                "correct_options, options_batch, and answer_texts must have the same "
                f"length; got {batch_lengths}."
            )
        return [
            format_mcq_target(
                correct_option=correct_option,
                options=options,
                answer_text=answer_text,
            )
            for correct_option, options, answer_text in zip(
                correct_options,
                options_batch,
                answer_texts,
            )
        ]

    def _semantic_attention_mask(
        self,
        mel_features: torch.Tensor,
        mel_mask: Optional[torch.Tensor],
        output_width: int,
    ) -> torch.Tensor:
        if mel_mask is None:
            mel_mask = torch.ones(
                (mel_features.shape[0], mel_features.shape[-1]),
                dtype=torch.bool,
                device=mel_features.device,
            )
        return self.backbone.build_audio_attention_mask(
            mel_mask,
            output_width=output_width,
        ).to(dtype=torch.bool)

    def _encode_fused_audio(
        self,
        batch: Dict,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        whisper_input_features = batch["whisper_input_features"]
        whisper_attention_mask = batch.get("whisper_attention_mask")
        with torch.no_grad():
            semantic_audio = self.backbone.encode_audio_features(
                whisper_input_features,
                mask=whisper_attention_mask,
                device=device,
            ).float()

        spatial_features = batch["spatial_features"].to(device)
        spatial_seq = self.spatial_encoder(spatial_features).float()
        semantic_attention_mask = self._semantic_attention_mask(
            mel_features=whisper_input_features,
            mel_mask=whisper_attention_mask,
            output_width=semantic_audio.shape[1],
        )
        return self.fusion(
            semantic_audio=semantic_audio,
            spatial_seq=spatial_seq,
            semantic_attention_mask=semantic_attention_mask,
        )

    def _build_layer1_prompt_ids(
        self,
        questions: Sequence[str],
        options_batch: Sequence[Dict[str, str]],
    ) -> List[torch.Tensor]:
        batch_lengths = (len(questions), len(options_batch))
        if len(set(batch_lengths)) != 1:
            raise ValueError(
                "questions and options_batch must have the same length; "
                f"got {batch_lengths}."
            )
        return [
            self.backbone.build_chat_prompt_ids(
                self.backbone.build_user_prompt(
                    question=question,
                    options=options,
                )
            )
            for question, options in zip(questions, options_batch)
        ]

    def _build_layer2_prompt_ids(
        self,
        layer1_questions: Sequence[str],
        layer1_answers: Sequence[str],
        layer2_questions: Sequence[str],
        layer1_options_batch: Sequence[Dict[str, str]],
        layer2_options_batch: Sequence[Dict[str, str]],
    ) -> List[torch.Tensor]:
        batch_lengths = (
            len(layer1_questions),
            len(layer1_answers),
            len(layer2_questions),
            len(layer1_options_batch),
            len(layer2_options_batch),
        )
        if len(set(batch_lengths)) != 1:
            raise ValueError(
                "Layer 1 questions, Layer 1 answers, Layer 2 questions, and both "
                f"options batches must have the same length; got {batch_lengths}."
            )

        prompt_ids_list: List[torch.Tensor] = []
        for question1, answer1, question2, options1, options2 in zip(
            layer1_questions,
            layer1_answers,
            layer2_questions,
            layer1_options_batch,
            layer2_options_batch,
        ):
            messages = [
                {
                    "role": "user",
                    "content": self.backbone.build_user_prompt(
                        question=question1,
                        options=options1,
                    ),
                },
                {
                    "role": "assistant",
                    "content": self._normalize_history_text(answer1),
                },
                {
                    "role": "user",
                    "content": self._build_followup_user_prompt(
                        question=question2,
                        options=options2,
                    ),
                },
            ]
            prompt_ids_list.append(
                self.backbone.build_chat_prompt_ids_from_messages(messages)
            )
        return prompt_ids_list

    def _prepare_lm_inputs(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        answers: Sequence[str],
        fused_audio_tokens: torch.Tensor,
        audio_content_mask: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        batch_lengths = (
            len(prompt_ids_list),
            len(answers),
            fused_audio_tokens.shape[0],
            audio_content_mask.shape[0],
        )
        if len(set(batch_lengths)) != 1:
            raise ValueError(
                "Prompt, answer, fused-audio, and audio-mask batch sizes must match; "
                f"got {batch_lengths}."
            )

        answer_ids_list = [
            self.backbone.tokenize_answer_ids(answer) for answer in answers
        ]
        return self.backbone.build_inputs_embeds(
            prompt_ids_list=prompt_ids_list,
            audio_tokens=fused_audio_tokens,
            spatial_tokens=None,
            audio_attention_mask=audio_content_mask,
            device=device,
            answer_ids_list=answer_ids_list,
            only_first_answer_token_loss=False,
        )

    def _layer_lm_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = (model_inputs["labels"] != -100).sum()
        if token_count.item() <= 0:
            raise ValueError("LM loss requires at least one supervised token.")
        outputs = self.backbone.llm(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
            use_cache=False,
            return_dict=True,
        )
        if not torch.isfinite(outputs.loss).all().item():
            raise FloatingPointError("LM loss is non-finite.")
        return outputs.loss, token_count

    def forward_train(
        self,
        batch: Dict,
        device: torch.device,
        enable_layer2: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not enable_layer2 and self.loss_weights["lm_layer1"] <= 0:
            raise ValueError(
                "lm_layer1 loss weight must be greater than zero when Layer 2 "
                "is disabled."
            )
        fuse = self._encode_fused_audio(batch, device=device)
        layer1_targets = self._format_mcq_targets(
            correct_options=batch["layer1_correct_option"],
            options_batch=batch["layer1_options"],
            answer_texts=batch["layer1_answers"],
        )
        layer1_inputs = self._prepare_lm_inputs(
            prompt_ids_list=self._build_layer1_prompt_ids(
                questions=batch["layer1_questions"],
                options_batch=batch["layer1_options"],
            ),
            answers=layer1_targets,
            fused_audio_tokens=fuse["fused_audio"],
            audio_content_mask=fuse["audio_content_mask"],
            device=device,
        )
        loss_layer1, tokens_layer1 = self._layer_lm_loss(layer1_inputs)

        if enable_layer2:
            layer2_targets = self._format_mcq_targets(
                correct_options=batch["layer2_correct_option"],
                options_batch=batch["layer2_options"],
                answer_texts=batch["layer2_answers"],
            )
            layer2_inputs = self._prepare_lm_inputs(
                prompt_ids_list=self._build_layer2_prompt_ids(
                    layer1_questions=batch["layer1_questions"],
                    layer1_answers=layer1_targets,
                    layer2_questions=batch["layer2_questions"],
                    layer1_options_batch=batch["layer1_options"],
                    layer2_options_batch=batch["layer2_options"],
                ),
                answers=layer2_targets,
                fused_audio_tokens=fuse["fused_audio"],
                audio_content_mask=fuse["audio_content_mask"],
                device=device,
            )
            loss_layer2, tokens_layer2 = self._layer_lm_loss(layer2_inputs)
            weighted_tokens_layer1 = tokens_layer1.to(
                dtype=loss_layer1.dtype
            ) * self.loss_weights["lm_layer1"]
            weighted_tokens_layer2 = tokens_layer2.to(
                dtype=loss_layer2.dtype
            ) * self.loss_weights["lm_layer2"]
            total_weight = weighted_tokens_layer1 + weighted_tokens_layer2
            if (
                not torch.isfinite(total_weight).all().item()
                or total_weight.item() <= 0
            ):
                raise ValueError(
                    "The total supervised weight must be finite and greater than zero."
                )
            total_loss = (
                loss_layer1 * weighted_tokens_layer1
                + loss_layer2 * weighted_tokens_layer2
            ) / total_weight
        else:
            total_loss = loss_layer1
            loss_layer2 = torch.zeros(
                (),
                device=device,
                dtype=loss_layer1.dtype,
            )

        mean_gate_layer1 = fuse["gate"].mean().detach()
        mean_gate_layer2 = (
            mean_gate_layer1
            if enable_layer2
            else torch.zeros((), device=device, dtype=mean_gate_layer1.dtype)
        )
        return {
            "loss": total_loss,
            "loss_next_token": total_loss.detach(),
            "loss_layer1_next_token": loss_layer1.detach(),
            "loss_layer2_next_token": loss_layer2.detach(),
            "mean_gate_layer1": mean_gate_layer1,
            "mean_gate_layer2": mean_gate_layer2,
        }

    def _build_generation_inputs(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        fused_audio_tokens: torch.Tensor,
        audio_content_mask: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        return self.backbone.build_inputs_embeds(
            prompt_ids_list=prompt_ids_list,
            audio_tokens=fused_audio_tokens,
            spatial_tokens=None,
            audio_attention_mask=audio_content_mask,
            device=device,
            answer_ids_list=None,
        )

    @torch.no_grad()
    def _generate_texts(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        fused_audio_tokens: torch.Tensor,
        audio_content_mask: torch.Tensor,
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> List[str]:
        model_inputs = self._build_generation_inputs(
            prompt_ids_list=prompt_ids_list,
            fused_audio_tokens=fused_audio_tokens,
            audio_content_mask=audio_content_mask,
            device=device,
        )
        return [
            text.strip()
            for text in self.backbone.generate_from_inputs_embeds(
                inputs_embeds=model_inputs["inputs_embeds"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        ]

    @torch.no_grad()
    def _generate_dual_layer_texts(
        self,
        batch: Dict,
        fused_audio_tokens: torch.Tensor,
        audio_content_mask: torch.Tensor,
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> tuple[List[str], List[str]]:
        layer1_prompt_ids = self._build_layer1_prompt_ids(
            questions=batch["layer1_questions"],
            options_batch=batch["layer1_options"],
        )
        layer1_text = self._generate_texts(
            prompt_ids_list=layer1_prompt_ids,
            fused_audio_tokens=fused_audio_tokens,
            audio_content_mask=audio_content_mask,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        layer1_history_answers = [
            self._normalize_history_text(text) for text in layer1_text
        ]
        layer2_prompt_ids = self._build_layer2_prompt_ids(
            layer1_questions=batch["layer1_questions"],
            layer1_answers=layer1_history_answers,
            layer2_questions=batch["layer2_questions"],
            layer1_options_batch=batch["layer1_options"],
            layer2_options_batch=batch["layer2_options"],
        )
        layer2_text = self._generate_texts(
            prompt_ids_list=layer2_prompt_ids,
            fused_audio_tokens=fused_audio_tokens,
            audio_content_mask=audio_content_mask,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return layer1_text, layer2_text

    @torch.no_grad()
    def generate_freeform_answers(
        self,
        batch: Dict,
        device: torch.device,
        max_new_tokens: int = 32,
        num_beams: int = 1,
    ) -> FreeformGenerationOutput:
        was_training = self.training
        try:
            self.eval()
            fuse = self._encode_fused_audio(batch, device=device)
            layer1_text, layer2_text = self._generate_dual_layer_texts(
                batch=batch,
                fused_audio_tokens=fuse["fused_audio"],
                audio_content_mask=fuse["audio_content_mask"],
                device=device,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            gate_mean = fuse["gate"].mean(dim=1).cpu().tolist()
            return {
                "layer1_text": layer1_text,
                "layer2_text": layer2_text,
                "layer1_gate": list(gate_mean),
                "layer2_gate": list(gate_mean),
            }
        finally:
            self.train(was_training)
