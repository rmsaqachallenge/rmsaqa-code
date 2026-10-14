"""Cache-only dataset and collator for MiDashengLM evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .midashenglm_cache import load_midasheng_cache
from .midashenglm_dataset import MiDashengSpatialQAPairDataset


class MiDashengEvalDataset(MiDashengSpatialQAPairDataset):
    """Evaluation dataset using normalized QA records and precomputed caches only."""

    EXPECTED_RAW_DIM = 1280
    EXPECTED_SPATIAL_SHAPE = (500, 448)
    def __init__(self, json_path, feature_root, expected_checkpoint_id=None,
                 expected_frontend_version="dasheng-64mel-v1", max_records=None,
                 task_id=None, online_waveform: bool = False,
                 extract_spatial: bool = False, audio_root=None):
        self.online_waveform = bool(online_waveform)
        self.extract_spatial = bool(extract_spatial)
        self.audio_root = Path(audio_root) if audio_root else None
        super().__init__(json_path, feature_root, expected_checkpoint_id,
                         expected_frontend_version, max_records, task_id)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        if self.online_waveform:
            from .feature_ops import AudioFeatureExtractor
            import torch.nn.functional as F
            audio_path = Path(record.audio_path)
            if self.audio_root and not audio_path.is_absolute(): audio_path = self.audio_root / audio_path
            wave = AudioFeatureExtractor().load_audio(audio_path)
            if wave.ndim == 1: wave = wave.unsqueeze(0)
            original_wave = wave
            import torchaudio
            wave = torchaudio.functional.resample(wave, 24000, 16000)
            valid = min(int(wave.shape[-1]), 160000)
            wave = wave[..., :160000]
            if wave.shape[-1] < 160000: wave = F.pad(wave, (0, 160000-wave.shape[-1]))
            item = {"input_values": wave[0].float(), "audio_length": valid}
            if self.extract_spatial: item["spatial_features"] = AudioFeatureExtractor().extract_spatial_features(original_wave).float()
            else: item["spatial_features"] = torch.zeros(self.EXPECTED_SPATIAL_SHAPE)
            # Continue below for QA metadata; cache fields are unavailable online.
            base = {"segment_id": record.segment_id, "audio_path": record.audio_path, "feature_source": "waveform", "feature_sources": {"dasheng": "waveform", "spatial": "waveform"}, "metadata": None}
            base.update(item)
            for k, v in (("layer1_question",record.layer1_question),("layer1_answer",record.layer1_answer),("layer2_question",record.layer2_question),("layer2_answer",record.layer2_answer),("layer1_options",record.layer1_options),("layer1_correct_option",record.layer1_correct_option),("layer2_options",record.layer2_options),("layer2_correct_option",record.layer2_correct_option),("layer2_category_id",record.layer2_category_id),("layer2_category_name",record.layer2_category_name),("task_id",record.task_id)): base[k]=v
            return base
        path = self._feature_path(record.segment_id)
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(f"Missing MiDasheng evaluation cache: {path}")
        loaded = load_midasheng_cache(path, self.expected_checkpoint_id,
                                      self.expected_frontend_version)
        # load_midasheng_cache performs strict metadata, dtype and shape checks.
        metadata = loaded["metadata"]
        if metadata.raw_dim != self.EXPECTED_RAW_DIM:
            raise ValueError(f"evaluation cache raw_dim must be {self.EXPECTED_RAW_DIM}")
        if (metadata.spatial_dim != self.EXPECTED_SPATIAL_SHAPE[1]
                or tuple(loaded["spatial_features"].shape) != self.EXPECTED_SPATIAL_SHAPE):
            raise ValueError(f"evaluation cache spatial_features must have shape {self.EXPECTED_SPATIAL_SHAPE}")
        source = {"dasheng": "cache", "spatial": "cache"}
        return {
            "segment_id": record.segment_id, "audio_path": record.audio_path,
            "dasheng_features": torch.as_tensor(loaded["dasheng_features"]).float(),
            "dasheng_attention_mask": torch.as_tensor(loaded["dasheng_attention_mask"]).long(),
            "spatial_features": torch.as_tensor(loaded["spatial_features"]).float(),
            "metadata": metadata, "feature_source": "cache", "feature_sources": source,
            "layer1_question": record.layer1_question, "layer1_answer": record.layer1_answer,
            "layer2_question": record.layer2_question, "layer2_answer": record.layer2_answer,
            "layer1_options": record.layer1_options, "layer1_correct_option": record.layer1_correct_option,
            "layer2_options": record.layer2_options, "layer2_correct_option": record.layer2_correct_option,
            "layer2_category_id": record.layer2_category_id,
            "layer2_category_name": record.layer2_category_name, "task_id": record.task_id,
        }


def collate_midashenglm_eval(batch: List[Dict]) -> Dict:
    if not batch:
        raise ValueError("batch must be non-empty")
    if "input_values" in batch[0]:
        out = {"input_values": torch.stack([item["input_values"] for item in batch]), "audio_length": [item["audio_length"] for item in batch],
               "spatial_features": torch.stack([item["spatial_features"] for item in batch])}
        for key, plural in (("segment_id", "segment_ids"), ("audio_path", "audio_paths"), ("layer1_question", "layer1_questions"), ("layer1_answer", "layer1_answers"), ("layer2_question", "layer2_questions"), ("layer2_answer", "layer2_answers"), ("layer1_options", "layer1_options"), ("layer1_correct_option", "layer1_correct_options"), ("layer2_options", "layer2_options"), ("layer2_correct_option", "layer2_correct_options"), ("layer2_category_id", "layer2_category_ids"), ("layer2_category_name", "layer2_category_names"), ("task_id", "task_ids"), ("feature_sources", "feature_sources"), ("metadata", "metadata")): out[plural] = [item.get(key) for item in batch]
        out["feature_source"] = [item.get("feature_source", "waveform") for item in batch]
        return out
    raw = [torch.as_tensor(item["dasheng_features"]) for item in batch]
    masks = [torch.as_tensor(item["dasheng_attention_mask"]) for item in batch]
    spatial = [torch.as_tensor(item["spatial_features"]) for item in batch]
    if any(x.ndim != 2 or m.ndim != 1 or x.shape[0] != m.shape[0] for x, m in zip(raw, masks)):
        raise ValueError("dasheng features and masks have incompatible shapes")
    if len({x.shape[1] for x in raw}) != 1:
        raise ValueError("dasheng features must have a consistent feature dimension")
    if any(x.ndim != 2 for x in spatial) or len({tuple(x.shape) for x in spatial}) != 1:
        raise ValueError("spatial features must be equally shaped matrices")
    max_t, dim = max(x.shape[0] for x in raw), raw[0].shape[1]
    raw_out = torch.zeros((len(batch), max_t, dim), dtype=torch.float32)
    mask_out = torch.zeros((len(batch), max_t), dtype=torch.long)
    for i, (x, m) in enumerate(zip(raw, masks)):
        raw_out[i, :x.shape[0]] = x.float(); mask_out[i, :x.shape[0]] = m.long()
    out = {"dasheng_features": raw_out, "dasheng_attention_mask": mask_out,
           "spatial_features": torch.stack([x.float() for x in spatial])}
    for key, plural in (("segment_id", "segment_ids"), ("audio_path", "audio_paths"),
                        ("layer1_question", "layer1_questions"), ("layer1_answer", "layer1_answers"),
                        ("layer2_question", "layer2_questions"), ("layer2_answer", "layer2_answers"),
                        ("layer1_options", "layer1_options"), ("layer1_correct_option", "layer1_correct_options"),
                        ("layer2_options", "layer2_options"), ("layer2_correct_option", "layer2_correct_options"),
                        ("layer2_category_id", "layer2_category_ids"), ("layer2_category_name", "layer2_category_names"),
                        ("task_id", "task_ids"), ("feature_sources", "feature_sources"), ("metadata", "metadata")):
        out[plural] = [item.get(key) for item in batch]
    out["feature_source"] = [item.get("feature_source", "cache") for item in batch]
    return out


# Short alias used by a few downstream callers.
collate_midasheng_eval = collate_midashenglm_eval
