from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..mcq import build_mcq_prompt_body, format_mcq_target
from .af3_generative_backbone import FrozenAF3GenerativeBackbone
from .spatial_encoder import ResnetConformerTokenEncoder


class TemporalConcatFusion(nn.Module):
    """
    Simple spatial plugin:
    1) spatial encoder outputs [B, T, spatial_dim]
    2) concatenate with semantic tokens [B, T, semantic_dim]
       (default semantic_dim == llm_dim for backward compatibility)
    3) project concatenated tokens back to AF3 token dim
    """

    def __init__(self, llm_dim: int, spatial_dim: int, semantic_dim: Optional[int] = None):
        super().__init__()
        self.llm_dim = llm_dim
        self.spatial_dim = spatial_dim
        self.semantic_dim = semantic_dim if semantic_dim is not None else llm_dim
        self.concat_project = nn.Linear(self.semantic_dim + spatial_dim, llm_dim)
        self._init_concat_project()

    def _init_concat_project(self) -> None:
        with torch.no_grad():
            self.concat_project.weight.zero_()
            self.concat_project.bias.zero_()
            if self.semantic_dim == self.llm_dim:
                # Identity-like: output equals semantic input when spatial path is zero.
                eye = torch.eye(self.llm_dim, dtype=self.concat_project.weight.dtype)
                self.concat_project.weight[:, : self.llm_dim].copy_(eye)
            else:
                # For raw semantic features (e.g. 1280-dim Whisper), exact identity
                # is impossible because input/output dims differ. Initialize the
                # semantic submatrix with Kaiming uniform and keep spatial submatrix
                # zero so the model starts from a semantic-only baseline.
                nn.init.kaiming_uniform_(self.concat_project.weight[:, : self.semantic_dim], a=math.sqrt(5))

    def forward(self, semantic_audio: torch.Tensor, spatial_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        if semantic_audio.size(-1) != self.semantic_dim:
            raise ValueError(
                f"Expected semantic_audio dim={self.semantic_dim}, but got {semantic_audio.size(-1)}."
            )
        if spatial_seq.size(1) != semantic_audio.size(1):
            raise ValueError(
                f"Temporal mismatch: spatial_seq has T={spatial_seq.size(1)}, "
                f"but semantic tokens have T={semantic_audio.size(1)}."
            )
        fused_audio = self.concat_project(torch.cat([semantic_audio, spatial_seq], dim=-1))
        pseudo_gate = torch.ones(
            semantic_audio.size(0),
            semantic_audio.size(1),
            dtype=semantic_audio.dtype,
            device=semantic_audio.device,
        )
        return {
            "fused_audio": fused_audio,
            "gate": pseudo_gate,
        }


class SpatialAF3GenerativeModel(nn.Module):
    def __init__(
        self,
        af3_root: str,
        audio_token_stride: int = 1,
        apply_stage35: bool = True,
        enable_llm_lora: bool = True,
        llm_lora_r: int = 16,
        llm_lora_alpha: int = 16,
        llm_lora_dropout: float = 0.05,
        llm_lora_num_layers: int = 28,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.backbone = FrozenAF3GenerativeBackbone(
            af3_root=af3_root,
            apply_stage35=apply_stage35,
            audio_token_stride=audio_token_stride,
            enable_llm_lora=enable_llm_lora,
            llm_lora_r=llm_lora_r,
            llm_lora_alpha=llm_lora_alpha,
            llm_lora_dropout=llm_lora_dropout,
            llm_lora_num_layers=llm_lora_num_layers,
        )
        self.spatial_encoder = ResnetConformerTokenEncoder(encoder_dim=1024)
        self.fusion = TemporalConcatFusion(
            llm_dim=self.backbone.hidden_size,
            spatial_dim=1024,
            semantic_dim=self.backbone.audio_hidden_size,
        )
        self.loss_weights = {
            "lm_stage1": 1.0,
            "lm_stage2": 1.0,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

    def train(self, mode: bool = True) -> "SpatialAF3GenerativeModel":
        super().train(mode)
        self.backbone.train(mode)
        return self

    @staticmethod
    def _normalize_history_text(text: str) -> str:
        text = (text or "").strip()
        return " ".join(text.split())

    def _build_followup_user_prompt(self, question: str, options: Dict[str, str]) -> str:
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
        return [
            format_mcq_target(correct_option=correct_option, options=options, answer_text=answer_text)
            for correct_option, options, answer_text in zip(correct_options, options_batch, answer_texts)
        ]

    def _build_stage1_prompt_ids(
        self,
        questions: Sequence[str],
        options_batch: Sequence[Dict[str, str]],
    ) -> List[torch.Tensor]:
        return [
            self.backbone.build_chat_prompt_ids(
                self.backbone.build_user_prompt(question=question, options=options)
            )
            for question, options in zip(questions, options_batch)
        ]

    def _build_stage2_prompt_ids(
        self,
        stage1_questions: Sequence[str],
        stage1_answers: Sequence[str],
        stage2_questions: Sequence[str],
        stage1_options_batch: Sequence[Dict[str, str]],
        stage2_options_batch: Sequence[Dict[str, str]],
    ) -> List[torch.Tensor]:
        prompt_ids_list: List[torch.Tensor] = []
        for question1, answer1, question2, options1, options2 in zip(
            stage1_questions,
            stage1_answers,
            stage2_questions,
            stage1_options_batch,
            stage2_options_batch,
        ):
            messages = [
                {
                    "role": "user",
                    "content": self.backbone.build_user_prompt(question=question1, options=options1),
                },
                {
                    "role": "assistant",
                    "content": self._normalize_history_text(answer1),
                },
                {
                    "role": "user",
                    "content": self._build_followup_user_prompt(question=question2, options=options2),
                },
            ]
            prompt_ids_list.append(self.backbone.build_chat_prompt_ids_from_messages(messages))
        return prompt_ids_list

    def _prepare_lm_inputs(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        answers: Sequence[str],
        fused_audio_tokens: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        answer_ids_list = [self.backbone.tokenize_answer_ids(ans) for ans in answers]
        return self.backbone.build_inputs_embeds(
            prompt_ids_list=prompt_ids_list,
            audio_tokens=fused_audio_tokens.to(dtype=self.backbone.text_dtype),
            spatial_tokens=None,
            device=device,
            answer_ids_list=answer_ids_list,
            only_first_answer_token_loss=False,
        )

    def _layer_lm_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone.llm(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
            use_cache=False,
            return_dict=True,
        )
        token_count = (model_inputs["labels"] != -100).sum()
        return torch.nan_to_num(outputs.loss), token_count

    def _encode_fused_audio(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        whisper_input_features = batch["whisper_input_features"]
        spatial_features = batch["spatial_features"].to(device)
        with torch.no_grad():
            semantic_audio = self.backbone.encode_audio_features(whisper_input_features, device=device).float()
        spatial_seq = self.spatial_encoder(spatial_features).float()
        return self.fusion(semantic_audio=semantic_audio, spatial_seq=spatial_seq)

    def forward_train(self, batch: Dict, device: torch.device, enable_stage2: bool = True) -> Dict[str, torch.Tensor]:
        fuse = self._encode_fused_audio(batch, device=device)

        stage1_targets = self._format_mcq_targets(
            correct_options=batch["stage1_correct_option"],
            options_batch=batch["stage1_options"],
            answer_texts=batch["stage1_answers"],
        )
        stage2_targets = self._format_mcq_targets(
            correct_options=batch["stage2_correct_option"],
            options_batch=batch["stage2_options"],
            answer_texts=batch["stage2_answers"],
        )

        stage1_inputs = self._prepare_lm_inputs(
            prompt_ids_list=self._build_stage1_prompt_ids(
                questions=batch["stage1_questions"],
                options_batch=batch["stage1_options"],
            ),
            answers=stage1_targets,
            fused_audio_tokens=fuse["fused_audio"],
            device=device,
        )
        loss_stage1, tokens_stage1 = self._layer_lm_loss(stage1_inputs)

        if enable_stage2:
            stage2_inputs = self._prepare_lm_inputs(
                prompt_ids_list=self._build_stage2_prompt_ids(
                    stage1_questions=batch["stage1_questions"],
                    stage1_answers=stage1_targets,
                    stage2_questions=batch["stage2_questions"],
                    stage1_options_batch=batch["stage1_options"],
                    stage2_options_batch=batch["stage2_options"],
                ),
                answers=stage2_targets,
                fused_audio_tokens=fuse["fused_audio"],
                device=device,
            )
            loss_stage2, tokens_stage2 = self._layer_lm_loss(stage2_inputs)
            weighted_tokens_stage1 = tokens_stage1.to(dtype=loss_stage1.dtype) * self.loss_weights["lm_stage1"]
            weighted_tokens_stage2 = tokens_stage2.to(dtype=loss_stage2.dtype) * self.loss_weights["lm_stage2"]
            total_weight = (weighted_tokens_stage1 + weighted_tokens_stage2).clamp_min(1.0)
            total_loss = (
                loss_stage1 * weighted_tokens_stage1
                + loss_stage2 * weighted_tokens_stage2
            ) / total_weight
        else:
            total_loss = loss_stage1
            loss_stage2 = torch.tensor(0.0, device=device)

        mean_gate = fuse["gate"].mean().detach()
        return {
            "loss": total_loss,
            "loss_next_token": total_loss.detach(),
            "loss_stage1_next_token": loss_stage1.detach(),
            "loss_stage2_next_token": loss_stage2.detach() if enable_stage2 else torch.tensor(0.0, device=device),
            "mean_gate_stage1": mean_gate,
            "mean_gate_stage2": mean_gate,
        }

    def _build_generation_inputs(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        fused_audio_tokens: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        return self.backbone.build_inputs_embeds(
            prompt_ids_list=prompt_ids_list,
            audio_tokens=fused_audio_tokens.to(dtype=self.backbone.text_dtype),
            spatial_tokens=None,
            device=device,
            answer_ids_list=None,
        )

    @torch.no_grad()
    def _generate_texts(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        fused_audio_tokens: torch.Tensor,
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> List[str]:
        model_inputs = self._build_generation_inputs(
            prompt_ids_list=prompt_ids_list,
            fused_audio_tokens=fused_audio_tokens,
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
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> tuple[List[str], List[str]]:
        stage1_prompt_ids = self._build_stage1_prompt_ids(
            questions=batch["stage1_questions"],
            options_batch=batch["stage1_options"],
        )
        stage1_text = self._generate_texts(
            prompt_ids_list=stage1_prompt_ids,
            fused_audio_tokens=fused_audio_tokens,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        stage1_history_answers = [self._normalize_history_text(text) for text in stage1_text]
        stage2_prompt_ids = self._build_stage2_prompt_ids(
            stage1_questions=batch["stage1_questions"],
            stage1_answers=stage1_history_answers,
            stage2_questions=batch["stage2_questions"],
            stage1_options_batch=batch["stage1_options"],
            stage2_options_batch=batch["stage2_options"],
        )
        stage2_text = self._generate_texts(
            prompt_ids_list=stage2_prompt_ids,
            fused_audio_tokens=fused_audio_tokens,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return stage1_text, stage2_text

    @torch.no_grad()
    def generate_freeform_answers(
        self,
        batch: Dict,
        device: torch.device,
        max_new_tokens: int = 32,
        num_beams: int = 1,
    ) -> Dict[str, List[str]]:
        self.eval()
        fuse = self._encode_fused_audio(batch, device=device)
        gen1, gen2 = self._generate_dual_layer_texts(
            batch=batch,
            fused_audio_tokens=fuse["fused_audio"],
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        gate_mean = fuse["gate"].mean(dim=1).cpu().tolist()
        return {
            "stage1_text": gen1,
            "stage2_text": gen2,
            "stage1_gate": gate_mean,
            "stage2_gate": gate_mean,
        }

