from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2ForCausalLM, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from ..mcq import build_mcq_prompt_body


def enable_non_reentrant_gradient_checkpointing(model: nn.Module) -> None:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


class SoundProjector(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden_dim: int = 3584, out_dim: int = 3584):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FrozenAF3GenerativeBackbone(nn.Module):
    SEMANTIC_MEL_FRAMES = 1000
    SEMANTIC_TOKEN_LENGTH = 500

    def __init__(
        self,
        af3_root: str | Path,
        apply_stage35: bool = True,
        text_dtype: torch.dtype = torch.bfloat16,
        audio_dtype: torch.dtype = torch.float16,
        audio_token_stride: int = 1,
        enable_llm_lora: bool = True,
        llm_lora_r: int = 16,
        llm_lora_alpha: int = 16,
        llm_lora_dropout: float = 0.05,
        llm_lora_num_layers: int = 28,
        llm_lora_target_modules: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        af3_root = Path(af3_root)
        self.af3_root = af3_root
        self.text_dtype = text_dtype
        self.audio_dtype = audio_dtype
        # Retained only for backward compatibility with older call sites.
        # The semantic audio path now uses the first 1000 mel frames only,
        # and Whisper's built-in stride-2 convolution maps them to 500 tokens.
        self.audio_token_stride = max(1, int(audio_token_stride))
        self.enable_llm_lora = bool(enable_llm_lora)
        self.llm_lora_r = max(0, int(llm_lora_r))
        self.llm_lora_alpha = int(llm_lora_alpha)
        self.llm_lora_dropout = float(llm_lora_dropout)
        self.llm_lora_num_layers = max(0, int(llm_lora_num_layers))
        self.llm_lora_target_modules = tuple(llm_lora_target_modules or ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.llm_has_trainable_lora = False

        self.tokenizer = AutoTokenizer.from_pretrained(af3_root / "llm", trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "[PAD]"
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.sound_token_id = self.tokenizer.convert_tokens_to_ids("<sound>")
        if self.sound_token_id is None:
            raise ValueError("Tokenizer does not contain <sound> special token.")

        self.llm = Qwen2ForCausalLM.from_pretrained(
            af3_root / "llm",
            torch_dtype=text_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        self.llm.config.use_cache = False
        enable_non_reentrant_gradient_checkpointing(self.llm)

        self.audio_encoder = self._load_whisper_encoder(af3_root / "sound_tower", dtype=audio_dtype)
        self.sound_projector = SoundProjector()
        projector_state = load_file(str(af3_root / "sound_mm_projector" / "model.safetensors"))
        self.sound_projector.load_state_dict(projector_state, strict=True)
        self.sound_projector.to(dtype=text_dtype)

        if apply_stage35 and (af3_root / "stage35" / "adapter_model.safetensors").exists():
            self._apply_stage35_lora(af3_root / "stage35")
        if self.enable_llm_lora and self.llm_lora_r > 0:
            self._inject_trainable_llm_lora()
        self.llm.config.use_cache = False
        enable_non_reentrant_gradient_checkpointing(self.llm)

        self.hidden_size = self.llm.config.hidden_size
        self.audio_hidden_size = self.audio_encoder.config.d_model
        self.freeze()
        self.train(False)

    def _load_whisper_encoder(self, sound_tower_dir: Path, dtype: torch.dtype) -> nn.Module:
        config = WhisperConfig.from_pretrained(sound_tower_dir)
        config.max_source_positions = self.SEMANTIC_TOKEN_LENGTH
        encoder = WhisperEncoder(config)
        encoder_state = load_file(str(sound_tower_dir / "model.safetensors"))
        if "embed_positions.weight" in encoder_state:
            encoder_state["embed_positions.weight"] = encoder_state["embed_positions.weight"][: self.SEMANTIC_TOKEN_LENGTH]
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"[FrozenAF3GenerativeBackbone] Whisper encoder missing keys: {len(missing)}")
        if unexpected:
            print(f"[FrozenAF3GenerativeBackbone] Whisper encoder unexpected keys: {len(unexpected)}")
        encoder.to(dtype=dtype)
        return encoder

    def _prepare_audio_features(
        self,
        whisper_input_features: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if whisper_input_features.size(-1) < self.SEMANTIC_MEL_FRAMES:
            raise ValueError(
                "whisper_input_features must have at least "
                f"{self.SEMANTIC_MEL_FRAMES} frames, but got {whisper_input_features.size(-1)}."
            )
        feats = whisper_input_features[..., : self.SEMANTIC_MEL_FRAMES]
        return feats.to(device=device, dtype=dtype)

    def _apply_stage35_lora(self, stage35_dir: Path) -> None:
        import json

        raw_cfg = json.loads((stage35_dir / "adapter_config.json").read_text())
        allowed_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        filtered_cfg = {k: v for k, v in raw_cfg.items() if k in allowed_keys}
        peft_cfg = LoraConfig(**filtered_cfg)
        llm_with_lora = get_peft_model(self.llm, peft_cfg)
        adapter_state = load_file(str(stage35_dir / "adapter_model.safetensors"))
        mapped_state = {}
        for key, value in adapter_state.items():
            new_key = key.replace("base_model.model.llm.model.", "base_model.model.model.")
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            mapped_state[new_key] = value

        non_lora_path = stage35_dir / "non_lora_trainables.bin"
        if non_lora_path.exists():
            non_lora = torch.load(non_lora_path, map_location="cpu")
            for key, value in non_lora.items():
                new_key = key.replace("base_model.model.llm.model.", "base_model.model.model.")
                mapped_state[new_key] = value

        missing, unexpected = llm_with_lora.load_state_dict(mapped_state, strict=False)
        print(f"[FrozenAF3GenerativeBackbone] loaded stage35 lora, missing={len(missing)} unexpected={len(unexpected)}")
        self.llm = llm_with_lora.merge_and_unload()
        self.llm.to(dtype=self.text_dtype)

    def _inject_trainable_llm_lora(self) -> None:
        num_hidden_layers = int(getattr(self.llm.config, "num_hidden_layers", 0))
        if num_hidden_layers <= 0:
            raise ValueError("LLM config does not expose a valid num_hidden_layers.")
        target_num_layers = min(self.llm_lora_num_layers, num_hidden_layers)
        if target_num_layers <= 0:
            return
        peft_cfg = LoraConfig(
            r=self.llm_lora_r,
            lora_alpha=self.llm_lora_alpha,
            lora_dropout=self.llm_lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(self.llm_lora_target_modules),
            layers_to_transform=list(range(target_num_layers)),
            layers_pattern="layers",
        )
        self.llm = get_peft_model(self.llm, peft_cfg)
        self.llm_has_trainable_lora = True
        print(
            "[FrozenAF3GenerativeBackbone] enabled LLM LoRA: "
            f"layers=0-{target_num_layers - 1}, target_modules={list(self.llm_lora_target_modules)}, r={self.llm_lora_r}"
        )

    @staticmethod
    def _is_lora_parameter(name: str) -> bool:
        return ".lora_" in name or name.startswith("lora_") or ".lora_embedding_" in name

    def freeze(self) -> None:
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.sound_projector.parameters():
            param.requires_grad = False
        for _, param in self.llm.named_parameters():
            param.requires_grad = False
        if self.llm_has_trainable_lora:
            for name, param in self.llm.named_parameters():
                if self._is_lora_parameter(name):
                    param.requires_grad = True

    def train(self, mode: bool = True) -> "FrozenAF3GenerativeBackbone":
        super().train(mode)
        self.freeze()
        self.audio_encoder.eval()
        self.sound_projector.eval()
        if self.llm_has_trainable_lora and mode:
            self.llm.train(True)
        else:
            self.llm.eval()
        return self

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def build_user_prompt(self, question: str, options: Optional[Dict[str, str]] = None) -> str:
        question = question.strip()
        header = (
            "<sound>\n"
            "You are given a first-order ambisonics audio clip. Use the audio content faithfully, and use spatial cues only when the question requires them.\n"
        )
        if options is not None:
            return header + build_mcq_prompt_body(question=question, options=options)
        body = f"Question: {question}\nAnswer briefly and directly."
        return header + body

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
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        return prompt_ids

    def build_chat_prompt_ids(self, user_prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": user_prompt}]
        return self.build_chat_prompt_ids_from_messages(messages)

    def tokenize_answer_ids(self, answer_text: str) -> torch.Tensor:
        answer = answer_text.strip() + self.tokenizer.eos_token
        return self.tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids[0]

    @torch.no_grad()
    def encode_audio_features(self, whisper_input_features: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Return raw Whisper encoder features (before SoundProjector), shape [B, T, 1280]."""
        audio_device = next(self.audio_encoder.parameters()).device
        target_device = device if device is not None else next(self.get_input_embeddings().parameters()).device

        feats = self._prepare_audio_features(
            whisper_input_features,
            device=audio_device,
            dtype=self.audio_dtype,
        )
        hidden = self.audio_encoder(input_features=feats).last_hidden_state
        if hidden.size(1) != self.SEMANTIC_TOKEN_LENGTH:
            raise ValueError(
                f"Expected semantic audio length {self.SEMANTIC_TOKEN_LENGTH}, but got {hidden.size(1)}."
            )
        return hidden.to(device=target_device, dtype=torch.float32)

    @torch.no_grad()
    def encode_audio_tokens(self, whisper_input_features: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        audio_device = next(self.audio_encoder.parameters()).device
        projector_device = next(self.sound_projector.parameters()).device
        llm_device = device if device is not None else next(self.get_input_embeddings().parameters()).device

        feats = self._prepare_audio_features(
            whisper_input_features,
            device=audio_device,
            dtype=self.audio_dtype,
        )
        hidden = self.audio_encoder(input_features=feats).last_hidden_state
        if hidden.size(1) != self.SEMANTIC_TOKEN_LENGTH:
            raise ValueError(
                f"Expected semantic audio length {self.SEMANTIC_TOKEN_LENGTH}, but got {hidden.size(1)}."
            )
        hidden = hidden.to(device=projector_device, dtype=self.text_dtype)
        tokens = self.sound_projector(hidden).to(device=llm_device, dtype=self.text_dtype)
        return tokens

    @torch.no_grad()
    def embed_token_ids(self, token_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        if token_ids.numel() == 0:
            return torch.zeros(0, self.hidden_size, device=device, dtype=self.text_dtype)
        embeds = self.get_input_embeddings()(token_ids.to(device))
        return embeds.to(dtype=self.text_dtype)

    def build_inputs_embeds(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
        audio_tokens: torch.Tensor,
        spatial_tokens: Optional[torch.Tensor],
        device: torch.device,
        answer_ids_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        only_first_answer_token_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        seq_embeds: List[torch.Tensor] = []
        seq_labels: List[torch.Tensor] = []
        seq_masks: List[torch.Tensor] = []
        max_len = 0

        for idx, prompt_ids in enumerate(prompt_ids_list):
            prompt_ids = prompt_ids.to(device)
            sound_pos = (prompt_ids == self.sound_token_id).nonzero(as_tuple=False).flatten()
            if sound_pos.numel() == 0:
                raise ValueError("Prompt does not contain <sound> token.")
            sound_pos = int(sound_pos[0].item())
            pre_ids = prompt_ids[:sound_pos]
            post_ids = prompt_ids[sound_pos + 1 :]

            pre_emb = self.embed_token_ids(pre_ids, device=device)
            post_emb = self.embed_token_ids(post_ids, device=device)
            mm_emb = audio_tokens[idx]
            if spatial_tokens is not None:
                mm_emb = torch.cat([mm_emb, spatial_tokens[idx]], dim=0)
            mm_emb = mm_emb.to(dtype=self.text_dtype)

            parts = [pre_emb, mm_emb, post_emb]
            labels = [
                torch.full((pre_emb.size(0),), -100, dtype=torch.long, device=device),
                torch.full((mm_emb.size(0),), -100, dtype=torch.long, device=device),
                torch.full((post_emb.size(0),), -100, dtype=torch.long, device=device),
            ]

            if answer_ids_list is not None and answer_ids_list[idx] is not None:
                answer_ids = answer_ids_list[idx].to(device)
                answer_emb = self.embed_token_ids(answer_ids, device=device)
                parts.append(answer_emb)
                if only_first_answer_token_loss:
                    answer_labels = torch.full_like(answer_ids, -100, dtype=torch.long, device=device)
                    if answer_ids.numel() > 0:
                        answer_labels[0] = answer_ids[0].long()
                    labels.append(answer_labels)
                else:
                    labels.append(answer_ids.long())

            embeds = torch.cat(parts, dim=0)
            label = torch.cat(labels, dim=0)
            attn = torch.ones(embeds.size(0), dtype=torch.long, device=device)
            seq_embeds.append(embeds)
            seq_labels.append(label)
            seq_masks.append(attn)
            max_len = max(max_len, embeds.size(0))

        batch_size = len(seq_embeds)
        embed_dim = seq_embeds[0].size(-1)
        padded_embeds = torch.zeros(batch_size, max_len, embed_dim, dtype=self.text_dtype, device=device)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

        for i, (embeds, mask, labels) in enumerate(zip(seq_embeds, seq_masks, seq_labels)):
            length = embeds.size(0)
            padded_embeds[i, :length] = embeds
            padded_mask[i, :length] = mask
            padded_labels[i, :length] = labels

        return {
            "inputs_embeds": padded_embeds,
            "attention_mask": padded_mask,
            "labels": padded_labels,
        }

    @torch.no_grad()
    def generate_from_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 12,
        temperature: float = 0.0,
        num_beams: int = 1,
    ) -> List[str]:
        llm = self.llm
        llm.eval()
        was_use_cache = llm.config.use_cache
        llm.config.use_cache = True
        if temperature and temperature > 0:
            do_sample = True
        else:
            do_sample = False
            temperature = 1.0
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(top_p=1.0, top_k=0)
        outputs = llm.generate(**gen_kwargs)
        llm.config.use_cache = was_use_cache
        prompt_len = attention_mask.size(1)
        if outputs.ndim == 2 and outputs.size(1) > prompt_len:
            outputs = outputs[:, prompt_len:]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
