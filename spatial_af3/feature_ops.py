from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from transformers import WhisperFeatureExtractor


@dataclass
class AudioFeatureConfig:
    sample_rate: int = 24000
    mono_target_sr: int = 16000
    semantic_clip_length_sec: float = 10.0
    spatial_clip_length_sec: float = 10.0
    af3_audio_token_length: int = 1500
    audio_token_stride: int = 3
    spatial_n_fft: int = 1024
    spatial_win_length: int = 960
    spatial_hop_length: int = 480
    spatial_n_mels: int = 64
    whisper_n_mels: int = 128
    whisper_n_fft: int = 400
    whisper_hop_length: int = 160
    whisper_chunk_length: int = 10
    center: bool = True
    eps: float = 1e-8


class AudioFeatureExtractor:
    def __init__(self, cfg: Optional[AudioFeatureConfig] = None):
        self.cfg = cfg or AudioFeatureConfig()
        self._window = torch.hann_window(self.cfg.spatial_win_length)
        self.target_audio_tokens = self.cfg.af3_audio_token_length // max(1, int(self.cfg.audio_token_stride))
        self.target_spatial_samples = int(round(self.cfg.sample_rate * self.cfg.spatial_clip_length_sec))
        self.target_mono_samples = int(round(self.cfg.mono_target_sr * self.cfg.semantic_clip_length_sec))
        self.target_whisper_frames = int(round(self.cfg.whisper_chunk_length * self.cfg.mono_target_sr / self.cfg.whisper_hop_length))
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.cfg.spatial_n_fft // 2 + 1,
            f_min=0.0,
            f_max=self.cfg.sample_rate / 2,
            n_mels=self.cfg.spatial_n_mels,
            sample_rate=self.cfg.sample_rate,
            norm="slaney",
            mel_scale="htk",
        )
        self._mel_fb = mel_fb.T.contiguous()  # [M, F]
        self._whisper_fe = WhisperFeatureExtractor(
            feature_size=self.cfg.whisper_n_mels,
            sampling_rate=self.cfg.mono_target_sr,
            hop_length=self.cfg.whisper_hop_length,
            n_fft=self.cfg.whisper_n_fft,
            chunk_length=self.cfg.whisper_chunk_length,
            return_attention_mask=True,
        )

    def load_audio(self, audio_path: str | Path) -> torch.Tensor:
        wave, sr = sf.read(str(audio_path), always_2d=True)
        if sr != self.cfg.sample_rate:
            raise ValueError(f"Expected sample rate {self.cfg.sample_rate}, but got {sr} for {audio_path}")
        wave = torch.from_numpy(wave.T).float()  # [C, T]
        if wave.size(0) < 4:
            raise ValueError(f"Expected FOA 4-channel input, got shape {tuple(wave.shape)}")
        return wave[:4]

    @staticmethod
    def _pad_or_trim(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
        cur = waveform.size(-1)
        if cur == target_samples:
            return waveform
        if cur > target_samples:
            return waveform[..., :target_samples]
        pad = target_samples - cur
        return F.pad(waveform, (0, pad))

    @staticmethod
    def _resample_time_sequence(sequence: torch.Tensor, target_frames: int) -> torch.Tensor:
        # sequence: [T, D]
        if sequence.size(0) == target_frames:
            return sequence
        seq = sequence.transpose(0, 1).unsqueeze(0)
        seq = F.interpolate(seq, size=target_frames, mode="linear", align_corners=False)
        return seq.squeeze(0).transpose(0, 1).contiguous()

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        window = self._window.to(waveform.device, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=self.cfg.spatial_n_fft,
            hop_length=self.cfg.spatial_hop_length,
            win_length=self.cfg.spatial_win_length,
            window=window,
            center=self.cfg.center,
            return_complex=True,
        )

    def _mel_project(self, power_like: torch.Tensor) -> torch.Tensor:
        fb = self._mel_fb.to(power_like.device, dtype=power_like.dtype)
        # power_like: [F, T]
        return torch.einsum("mf,ft->mt", fb, power_like)

    def extract_spatial_features(self, foa_waveform: torch.Tensor) -> torch.Tensor:
        # Spatial branch is extracted directly from the 10 s FOA clip, then resampled to
        # the fusion token length used by the semantic branch (default: 500).
        foa_waveform = self._pad_or_trim(foa_waveform, self.target_spatial_samples)
        stft = self._stft(foa_waveform)  # [4, F, T]
        power = stft.abs().pow(2)
        w, x, y, z = stft[0], stft[1], stft[2], stft[3]
        denom = (power.sum(dim=0) + self.cfg.eps)

        logmel_channels = []
        for ch in range(4):
            mel = self._mel_project(power[ch])
            logmel_channels.append(torch.log(mel.clamp_min(self.cfg.eps)))

        ix = (torch.conj(w) * x).real / denom
        iy = (torch.conj(w) * y).real / denom
        iz = (torch.conj(w) * z).real / denom
        iv_channels = [torch.tanh(self._mel_project(v) * 5.0) for v in (ix, iy, iz)]

        feat = torch.cat(logmel_channels + iv_channels, dim=0).transpose(0, 1).contiguous()  # [T, 7*64]
        feat = self._resample_time_sequence(feat, target_frames=self.target_audio_tokens)
        return feat

    def extract_whisper_features(self, foa_waveform: torch.Tensor) -> torch.Tensor:
        mono = foa_waveform[0:1]
        if self.cfg.sample_rate != self.cfg.mono_target_sr:
            mono = torchaudio.functional.resample(mono, self.cfg.sample_rate, self.cfg.mono_target_sr)
        mono = self._pad_or_trim(mono, self.target_mono_samples)
        mono_np = mono.squeeze(0).cpu().numpy()
        input_features = self._whisper_fe(mono_np, sampling_rate=self.cfg.mono_target_sr, return_tensors="pt").input_features
        input_features = input_features.squeeze(0)
        if input_features.size(-1) != self.target_whisper_frames:
            raise ValueError(
                f"Expected whisper_input_features to have {self.target_whisper_frames} frames, "
                f"but got {input_features.size(-1)}."
            )
        return input_features  # [128, 1000]

    def extract_from_path(self, audio_path: str | Path) -> Dict[str, np.ndarray]:
        waveform = self.load_audio(audio_path)
        spatial = self.extract_spatial_features(waveform).cpu().numpy().astype(np.float16)
        whisper = self.extract_whisper_features(waveform).cpu().numpy().astype(np.float16)
        return {
            "spatial_features": spatial,
            "whisper_input_features": whisper,
        }


def save_feature_npz(data: Dict[str, np.ndarray], out_path: str | Path, compressed: bool = False) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(out_path, **data)
    else:
        np.savez(out_path, **data)


def load_feature_npz(path: str | Path) -> Dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {k: torch.from_numpy(v) for k, v in data.items()}
