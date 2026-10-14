from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..layered_mcq import format_mcq_target
from ..midashenglm_prompts import (
    build_followup_user_prompt,
    build_layer2_messages,
    normalize_history_text,
)
from .midashenglm_spatial_generative_backbone import (
    SpatialMiDashengLMGenerativeBackbone,
)
from .spatial_encoder import ResnetConformerTokenEncoder


class SpatialMiDashengLMTrainingModel(nn.Module):
    def __init__(self, midashenglm_root: str, enable_layer2: bool = True, **kwargs):
        super().__init__()
        self.enable_layer2 = enable_layer2
        self.spatial_encoder = ResnetConformerTokenEncoder(encoder_dim=1024)
        self.backbone = SpatialMiDashengLMGenerativeBackbone(midashenglm_root, **kwargs)
        self.audit_trainable_parameters()

    def trainable_parameter_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def audit_trainable_parameters(self):
        names = self.trainable_parameter_names()
        allowed = (
            "spatial_encoder.",
            "backbone.spatial_projector.",
            "backbone.backbone.decoder.",
            "backbone.backbone.llm.decoder.",
        )
        bad = [
            n
            for n in names
            if not n.startswith(allowed)
            or (
                n.startswith("backbone.backbone.decoder.")
                or n.startswith("backbone.backbone.llm.decoder.")
            )
            and not ("lora_A" in n or "lora_B" in n)
        ]
        if bad:
            raise RuntimeError(f"unexpected trainable parameters: {bad}")
        return names

    def train(self, mode=True):
        super().train(mode)
        self.backbone.train(mode)
        self.spatial_encoder.train(mode)
        return self

    @staticmethod
    def _format_mcq_targets(
        correct_options: Sequence[str],
        options_batch: Sequence[dict[str, str]],
        answer_texts: Sequence[str],
    ) -> list[str]:
        return [
            format_mcq_target(
                correct_option=correct_option,
                options=options,
                answer_text=answer_text,
            )
            for correct_option, options, answer_text in zip(
                correct_options, options_batch, answer_texts
            )
        ]

    @staticmethod
    def _combine_layer_losses(
        losses: Sequence[torch.Tensor], token_counts: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if not losses or len(losses) != len(token_counts):
            raise ValueError("losses and token_counts must be non-empty and aligned")
        dtype = losses[0].dtype
        weighted = torch.stack(
            [
                loss * count.to(device=loss.device, dtype=dtype)
                for loss, count in zip(losses, token_counts)
            ]
        )
        total_tokens = torch.stack(
            [count.to(device=losses[0].device, dtype=dtype) for count in token_counts]
        ).sum()
        return weighted.sum() / total_tokens.clamp_min(1.0)

    @staticmethod
    def _normalize_history_text(text: str) -> str:
        return normalize_history_text(text)

    def _build_followup_user_prompt(self, question: str, options: dict[str, str]) -> str:
        return build_followup_user_prompt(question, options)

    def _layer1_prompt_ids(self, batch, audio_length: int):
        return [
            self.backbone.build_chat_prompt_ids(
                self.backbone.build_user_prompt(question, options), audio_length
            )
            for question, options in zip(batch["layer1_question"], batch["layer1_options"])
        ]

    def _layer2_prompt_ids(self, batch, audio_length: int, layer1_answers: Sequence[str]):
        prompts = []
        for question1, options1, answer1, question2, options2 in zip(
            batch["layer1_question"],
            batch["layer1_options"],
            layer1_answers,
            batch["layer2_question"],
            batch["layer2_options"],
        ):
            messages = build_layer2_messages(
                self.backbone,
                layer1_question=question1,
                layer1_options=options1,
                layer1_answer=answer1,
                layer2_question=question2,
                layer2_options=options2,
            )
            prompts.append(
                self.backbone.build_chat_prompt_ids_from_messages(messages, audio_length)
            )
        return prompts

    def _tokenize_answers(self, answers: Sequence[str]) -> list[torch.Tensor]:
        result = []
        for answer in answers:
            token_ids = self.backbone.tokenizer(
                answer,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]
            if token_ids.numel() == 0:
                raise ValueError("tokenizer produced empty answer ids")
            if token_ids[-1].item() != self.backbone.eos_token_id:
                token_ids = torch.cat(
                    [
                        token_ids,
                        torch.tensor(
                            [self.backbone.eos_token_id],
                            dtype=token_ids.dtype,
                            device=token_ids.device,
                        ),
                    ]
                )
            result.append(token_ids)
        return result

    def forward(self, batch):
        spatial = self.spatial_encoder(batch["spatial_features"].float())
        cached = self.backbone.encode_cached_spatial_audio_bundle(
            batch["dasheng_features"], batch["dasheng_attention_mask"], spatial
        )
        for layer in (1, 2):
            questions = batch[f"layer{layer}_question"]
            options = batch[f"layer{layer}_options"]
            if options is None or len(options) != len(questions):
                raise ValueError("options batch length mismatch")
        layer1_targets = self._format_mcq_targets(
            batch["layer1_correct_option"],
            batch["layer1_options"],
            batch["layer1_answer"],
        )
        layer2_targets = self._format_mcq_targets(
            batch["layer2_correct_option"],
            batch["layer2_options"],
            batch["layer2_answer"],
        )
        losses = []
        token_counts = []
        out = {}
        for layer, targets in ((1, layer1_targets), (2, layer2_targets)):
            if layer == 2 and not self.enable_layer2:
                continue
            if layer == 1:
                prompt_ids = self._layer1_prompt_ids(batch, cached["spatial_tokens"].shape[1])
            else:
                prompt_ids = self._layer2_prompt_ids(
                    batch,
                    cached["spatial_tokens"].shape[1],
                    layer1_targets,
                )
            answer_ids = self._tokenize_answers(targets)
            model_inputs = self.backbone.build_inputs_embeds(
                prompt_ids,
                cached["spatial_tokens"],
                answer_ids,
                device=cached["spatial_tokens"].device,
            )
            result = self.backbone.forward_lm(model_inputs)
            losses.append(result.loss)
            token_counts.append((model_inputs["labels"] != -100).sum())
            out[f"layer{layer}_loss"] = result.loss
        out["loss"] = self._combine_layer_losses(losses, token_counts)
        return out

    @torch.no_grad()
    def generate_freeform_answers(
        self, batch, *, max_new_tokens: int = 12, num_beams: int = 1
    ):
        spatial = self.spatial_encoder(batch["spatial_features"].float())
        cached = self.backbone.encode_cached_spatial_audio_bundle(
            batch["dasheng_features"], batch["dasheng_attention_mask"], spatial
        )
        audio_length = cached["spatial_tokens"].shape[1]
        layer1_prompt_ids = self._layer1_prompt_ids(batch, audio_length)
        layer1_inputs = self.backbone.build_inputs_embeds(
            layer1_prompt_ids,
            cached["spatial_tokens"],
            device=cached["spatial_tokens"].device,
        )
        layer1_text = self.backbone.generate_from_inputs_embeds(
            layer1_inputs["inputs_embeds"],
            layer1_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        result = {"layer1_text": layer1_text}
        if self.enable_layer2:
            layer2_prompt_ids = self._layer2_prompt_ids(
                batch,
                audio_length,
                [self._normalize_history_text(text) for text in layer1_text],
            )
            layer2_inputs = self.backbone.build_inputs_embeds(
                layer2_prompt_ids,
                cached["spatial_tokens"],
                device=cached["spatial_tokens"].device,
            )
            result["layer2_text"] = self.backbone.generate_from_inputs_embeds(
                layer2_inputs["inputs_embeds"],
                layer2_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        return result
