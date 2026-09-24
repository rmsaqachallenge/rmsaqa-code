from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .feature_ops import AudioFeatureConfig, AudioFeatureExtractor, load_feature_npz, save_feature_npz


@dataclass
class QAPairRecord:
    segment_id: str
    audio_path: str
    stage1_question: str
    stage1_answer: str
    stage2_question: str
    stage2_answer: str
    stage1_options: Optional[Dict[str, str]] = None
    stage1_correct_option: Optional[str] = None
    stage2_options: Optional[Dict[str, str]] = None
    stage2_correct_option: Optional[str] = None
    stage2_category_id: Optional[int] = None
    stage2_category_name: Optional[str] = None
    task_id: Optional[int] = None


class SpatialQAPairDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        feature_root: Optional[str | Path] = None,
        extract_features_on_the_fly: bool = False,
        write_missing_features: bool = False,
        audio_token_stride: int = 3,
        max_records: Optional[int] = None,
        task_id: Optional[int] = None,
    ):
        self.json_path = Path(json_path)
        self.feature_root = Path(feature_root) if feature_root is not None else None
        self.extract_features_on_the_fly = extract_features_on_the_fly
        self.write_missing_features = write_missing_features
        self.audio_token_stride = max(1, int(audio_token_stride))
        self.task_id = task_id
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive when provided")
        cfg = AudioFeatureConfig(audio_token_stride=self.audio_token_stride)
        self.expected_spatial_frames = 1500 // self.audio_token_stride
        self.expected_whisper_frames = int(round(cfg.whisper_chunk_length * cfg.mono_target_sr / cfg.whisper_hop_length))
        self.records = self._load_records(max_records=max_records)
        self.extractor = (
            AudioFeatureExtractor(cfg)
            if extract_features_on_the_fly
            else None
        )

    def _load_records(self, max_records: Optional[int] = None) -> List[QAPairRecord]:
        records: List[QAPairRecord] = []
        payload_paths = self._payload_paths()
        per_sample_json = self.json_path.is_dir()
        for payload_path in payload_paths:
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Failed to load dataset JSON {payload_path}: {exc}") from exc

            payload_records = payload.get("records") if isinstance(payload, dict) else None
            if not isinstance(payload_records, list):
                raise ValueError(f"{payload_path} must contain a 'records' list")
            if per_sample_json and len(payload_records) != 1:
                raise ValueError(
                    f"Per-sample JSON {payload_path} must contain exactly one record"
                )

            for item in payload_records:
                record = self._normalize_record(item, payload_path)
                if self.task_id is not None and record.task_id != self.task_id:
                    continue
                records.append(record)
                if max_records is not None and len(records) >= max_records:
                    return records
        return records

    def _payload_paths(self) -> List[Path]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.json_path}")
        if self.json_path.is_file():
            if self.json_path.suffix.lower() != ".json":
                raise ValueError(f"Dataset file must be JSON: {self.json_path}")
            return [self.json_path]
        if not self.json_path.is_dir():
            raise ValueError(
                f"Dataset path must be a JSON file or directory: {self.json_path}"
            )

        paths = sorted(
            path
            for path in self.json_path.glob("*.json")
            if path.name.casefold() != "metadata.json"
        )
        if not paths:
            raise ValueError(f"Dataset directory contains no sample JSON files: {self.json_path}")
        return paths

    @staticmethod
    def _qa_stage(qa: Dict, payload_path: Path) -> int:
        stage = qa.get("stage", qa.get("layer"))
        if isinstance(stage, bool) or not isinstance(stage, int) or stage not in (1, 2):
            raise ValueError(f"QA in {payload_path} has invalid stage: {stage!r}")
        return stage

    @staticmethod
    def _normalize_answer(qa: Dict, payload_path: Path) -> tuple[str, Optional[str]]:
        if "answer" not in qa:
            raise ValueError(f"QA in {payload_path} is missing 'answer'")
        answer = qa["answer"]
        options = qa.get("options")
        correct_option = qa.get("correct_option")

        if "stage" in qa:
            if (
                not isinstance(options, dict)
                or not isinstance(answer, str)
                or answer not in options
            ):
                raise ValueError(
                    f"QA in {payload_path} has answer option {answer!r} "
                    "that is absent from options"
                )
            if correct_option is not None and str(correct_option) != answer:
                raise ValueError(
                    f"QA in {payload_path} has answer option {answer!r} that "
                    f"conflicts with correct_option {str(correct_option)!r}"
                )
            return str(options[answer]), answer

        if correct_option is not None:
            correct_option = str(correct_option)
            if isinstance(options, dict) and correct_option not in options:
                raise ValueError(
                    f"QA in {payload_path} has correct option {correct_option!r} "
                    "that is absent from options"
                )
        return str(answer), correct_option

    def _normalize_record(self, item: Dict, payload_path: Path) -> QAPairRecord:
        if not isinstance(item, dict):
            raise ValueError(f"Record in {payload_path} must be an object")
        segment_id = item.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            raise ValueError(f"Record in {payload_path} has invalid segment_id")
        qa_pairs = item.get("qa_pairs")
        if not isinstance(qa_pairs, list):
            raise ValueError(f"Record {segment_id} in {payload_path} has no qa_pairs list")

        by_stage: Dict[int, Dict] = {}
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                raise ValueError(f"Record {segment_id} in {payload_path} has a non-object QA")
            stage = self._qa_stage(qa, payload_path)
            if stage in by_stage:
                raise ValueError(f"Record {segment_id} in {payload_path} repeats stage {stage}")
            by_stage[stage] = qa
        if set(by_stage) != {1, 2}:
            raise ValueError(
                f"Expected exactly stages 1 and 2 in {segment_id}, found {sorted(by_stage)}"
            )

        q1, q2 = by_stage[1], by_stage[2]
        questions = []
        for stage, qa in ((1, q1), (2, q2)):
            question = qa.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Record {segment_id} in {payload_path}, stage {stage}, "
                    "has a missing or empty question"
                )
            questions.append(question)
        q1_question, q2_question = questions
        q1_answer, q1_correct_option = self._normalize_answer(q1, payload_path)
        q2_answer, q2_correct_option = self._normalize_answer(q2, payload_path)
        task_id = item.get("task_id")
        stage2_category_id = q2.get("category_id")
        if stage2_category_id is not None:
            raw_category_id = stage2_category_id
            if isinstance(raw_category_id, bool):
                raise ValueError(
                    f"Record {segment_id} in {payload_path} has invalid "
                    f"Stage-2 category_id {raw_category_id!r}"
                )
            if isinstance(raw_category_id, int):
                stage2_category_id = raw_category_id
            elif isinstance(raw_category_id, str) and raw_category_id.strip().isdigit():
                stage2_category_id = int(raw_category_id.strip())
            else:
                raise ValueError(
                    f"Record {segment_id} in {payload_path} has invalid "
                    f"Stage-2 category_id {raw_category_id!r}"
                )
            if stage2_category_id not in range(1, 7):
                raise ValueError(
                    f"Record {segment_id} in {payload_path} has unsupported "
                    f"Stage-2 category_id {stage2_category_id}"
            )
        if task_id is None and stage2_category_id is not None:
            task_id = 2 if stage2_category_id == 6 else 1
        audio_path = item.get("audio_path")
        if audio_path is None:
            audio_path = ""
        elif isinstance(audio_path, str):
            audio_path = audio_path.strip()
        else:
            raise ValueError(
                f"Record {segment_id} in {payload_path} has invalid audio_path "
                f"{audio_path!r}"
            )
        if not audio_path:
            audio_filename = Path(segment_id).with_suffix(".wav").name
            if segment_id.startswith("dev-"):
                audio_path = str(Path("dev_audio") / audio_filename)
            elif segment_id.startswith("train-"):
                audio_path = str(Path("train_audio") / audio_filename)

        return QAPairRecord(
            segment_id=segment_id,
            audio_path=audio_path,
            stage1_question=q1_question,
            stage1_answer=q1_answer,
            stage2_question=q2_question,
            stage2_answer=q2_answer,
            stage1_options=q1.get("options"),
            stage1_correct_option=q1_correct_option,
            stage2_options=q2.get("options"),
            stage2_correct_option=q2_correct_option,
            stage2_category_id=stage2_category_id,
            stage2_category_name=q2.get("category_name"),
            task_id=task_id,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _feature_path(self, segment_id: str) -> Optional[Path]:
        if self.feature_root is None:
            return None
        feature_id = segment_id[:-5] if segment_id.lower().endswith(".json") else segment_id
        return self.feature_root / f"{feature_id}.npz"

    def _load_or_extract_features(self, record: QAPairRecord) -> Dict[str, torch.Tensor]:
        feature_path = self._feature_path(record.segment_id)
        if feature_path is not None and feature_path.exists():
            feat = load_feature_npz(feature_path)
        else:
            if not self.extract_features_on_the_fly:
                raise FileNotFoundError(
                    f"Missing cached features for {record.segment_id}: {feature_path}. "
                    f"Run exp/preprocess_audio_features.py or enable --extract-features-on-the-fly."
                )
            assert self.extractor is not None
            if not record.audio_path:
                raise ValueError(
                    f"Record {record.segment_id} has no audio_path, so missing features "
                    "cannot be extracted on the fly. Precompute the cached feature file instead."
                )
            data = self.extractor.extract_from_path(record.audio_path)
            if feature_path is not None and self.write_missing_features:
                save_feature_npz(data, feature_path)
            feat = {k: torch.from_numpy(v) for k, v in data.items()}
        spatial_frames = int(feat["spatial_features"].shape[0])
        if spatial_frames != self.expected_spatial_frames:
            raise ValueError(
                f"Feature {record.segment_id} has spatial length {spatial_frames}, but expected "
                f"{self.expected_spatial_frames} for audio_token_stride={self.audio_token_stride}. "
                f"Please regenerate features with matching --audio-token-stride."
            )
        whisper_frames = int(feat["whisper_input_features"].shape[-1])
        if whisper_frames != self.expected_whisper_frames:
            raise ValueError(
                f"Feature {record.segment_id} has whisper length {whisper_frames}, but expected "
                f"{self.expected_whisper_frames}. Please regenerate cached features with the new 10 s / 1000-frame setting."
            )
        return feat

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        feat = self._load_or_extract_features(record)
        return {
            "segment_id": record.segment_id,
            "audio_path": record.audio_path,
            "spatial_features": feat["spatial_features"].float(),
            "whisper_input_features": feat["whisper_input_features"].float(),
            "stage1_question": record.stage1_question,
            "stage1_answer": record.stage1_answer,
            "stage2_question": record.stage2_question,
            "stage2_answer": record.stage2_answer,
            "stage1_options": record.stage1_options,
            "stage1_correct_option": record.stage1_correct_option,
            "stage2_options": record.stage2_options,
            "stage2_correct_option": record.stage2_correct_option,
            "stage2_category_id": record.stage2_category_id,
            "stage2_category_name": record.stage2_category_name,
            "task_id": record.task_id,
        }


def qa_pair_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "segment_ids": [item["segment_id"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
        "spatial_features": torch.stack([item["spatial_features"] for item in batch], dim=0),
        "whisper_input_features": torch.stack([item["whisper_input_features"] for item in batch], dim=0),
        "stage1_questions": [item["stage1_question"] for item in batch],
        "stage1_answers": [item["stage1_answer"] for item in batch],
        "stage2_questions": [item["stage2_question"] for item in batch],
        "stage2_answers": [item["stage2_answer"] for item in batch],
        "stage1_options": [item["stage1_options"] for item in batch],
        "stage1_correct_option": [item["stage1_correct_option"] for item in batch],
        "stage2_options": [item["stage2_options"] for item in batch],
        "stage2_correct_option": [item["stage2_correct_option"] for item in batch],
        "stage2_category_ids": [item["stage2_category_id"] for item in batch],
        "stage2_category_names": [item["stage2_category_name"] for item in batch],
        "task_ids": [item.get("task_id") for item in batch],
    }
