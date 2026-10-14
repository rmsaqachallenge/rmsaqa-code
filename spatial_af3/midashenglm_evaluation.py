"""MiDashengLM-specific MCQ evaluation and artifact writing."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch

from .layered_mcq import extract_mcq_option
from .metrics import CategoryHierarchicalAccuracyMeter, validate_stage2_category


def _field(batch, plural, singular):
    value = batch.get(plural, batch.get(singular))
    if value is None:
        raise ValueError(f"batch is missing {plural}")
    return value


@torch.inference_mode()
def evaluate_midashenglm_model(
    adapter, loader: Iterable[Mapping], device, max_new_tokens: int, num_beams: int,
    output_dir: Optional[str | Path] = None,
    answers_log_path: Optional[str | Path] = None,
):
    """Run deterministic MiDashengLM evaluation, returning ``(metrics, predictions)``."""
    if output_dir is not None:
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if answers_log_path is None and output_dir is not None:
        answers_log_path = output_dir / "answers_log.jsonl"
    log = Path(answers_log_path) if answers_log_path is not None else None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    meter, records = CategoryHierarchicalAccuracyMeter(), []
    with (log.open("w", encoding="utf-8") if log else nullcontext(None)) as stream:
        if hasattr(adapter, "eval"):
            adapter.eval()
        for batch in loader:
            ids = _field(batch, "segment_ids", "segment_id")
            n = len(ids)
            if n == 0:
                raise ValueError("batch segment_ids must be non-empty")
            required_fields = (
                "layer1_options", "layer2_options", "layer1_correct_options",
                "layer2_correct_options", "layer2_category_ids", "layer2_category_names",
                "audio_paths", "task_ids", "layer1_questions", "layer2_questions",
            )
            for key in required_fields:
                if key not in batch or len(batch[key]) != n:
                    raise ValueError(f"batch field {key!r}: expected length {n}")
            source_values = batch.get("feature_source", batch.get("feature_sources"))
            if source_values is None:
                raise ValueError("batch must include feature_source(s)")
            if not isinstance(source_values, str) and len(source_values) != n:
                raise ValueError(f"batch field 'feature_source': expected length {n}")
            source_check = [source_values] * n if isinstance(source_values, str) else source_values
            for src in source_check:
                if src not in {"cache", "waveform"}:
                    raise ValueError(f"invalid feature_source: {src!r}")
            for i in range(n):
                for option_key, ref_key in (("layer1_options", "layer1_correct_options"),
                                            ("layer2_options", "layer2_correct_options")):
                    options, ref = batch[option_key][i], batch[ref_key][i]
                    if not isinstance(options, dict) or not options:
                        raise ValueError(f"{option_key} must contain non-empty dicts")
                    if ref is None or ref not in options:
                        raise ValueError(f"{ref_key} must reference an option")
            try:
                outputs = adapter.generate_freeform_answers(
                    batch, device=device, max_new_tokens=max_new_tokens, num_beams=num_beams)
            except Exception as exc:
                raise RuntimeError(
                    f"MiDashengLM generation failed for segment_id={ids[0]!r}: {exc}"
                ) from exc
            for key in ("layer1_text", "layer2_text"):
                if key not in outputs or len(outputs[key]) != n:
                    raise ValueError(f"outputs field {key!r}: expected length {n}")
            if "audio_token_count" not in outputs:
                raise ValueError("outputs field 'audio_token_count' is required")
            counts = outputs["audio_token_count"]
            if len(counts) != n: raise ValueError(f"outputs field 'audio_token_count': expected length {n}")
            for i, segment_id in enumerate(ids):
                if not isinstance(outputs["layer1_text"][i], str) or not isinstance(outputs["layer2_text"][i], str):
                    raise ValueError("generated texts must be strings")
                o1 = _field(batch, "layer1_options", "layer1_options")[i]
                o2 = _field(batch, "layer2_options", "layer2_options")[i]
                r1 = _field(batch, "layer1_correct_options", "layer1_correct_option")[i]
                r2 = _field(batch, "layer2_correct_options", "layer2_correct_option")[i]
                c_id = batch["layer2_category_ids"][i]
                c_name = batch["layer2_category_names"][i]
                validate_stage2_category(c_id, c_name)
                p1, s1 = extract_mcq_option(outputs["layer1_text"][i], o1)
                p2, s2 = extract_mcq_option(outputs["layer2_text"][i], o2)
                c1, c2 = p1 == r1, p2 == r2
                meter.update(category_id=c_id, category_name=c_name, segment_id=segment_id,
                             stage1_correct=c1, stage2_correct=c2, stage1_pred=p1, stage1_ref=r1,
                             stage2_pred=p2, stage2_ref=r2)
                count = counts[i]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"audio_token_count for segment_id={segment_id!r} must be positive")
                rec = {
                    "segment_id": segment_id,
                    "audio_path": batch.get("audio_paths", [None] * n)[i],
                    "task_id": batch.get("task_ids", [None] * n)[i],
                    "stage2_category_id": c_id,
                    "stage2_category_name": c_name,
                    "stage1_pred_option": p1,
                    "stage1_pred_option_text": o1.get(p1, "") if p1 is not None else "",
                    "stage1_extract_status": s1,
                    "stage1_ref_option": r1,
                    "stage1_ref_option_text": o1[r1],
                    "stage1_correct": c1,
                    "stage2_pred_option": p2,
                    "stage2_pred_option_text": o2.get(p2, "") if p2 is not None else "",
                    "stage2_extract_status": s2,
                    "stage2_ref_option": r2,
                    "stage2_ref_option_text": o2[r2],
                    "stage2_correct_raw": c2,
                    "grounded_stage2_correct": bool(c1 and c2),
                    "stage1_question": batch["layer1_questions"][i],
                    "stage2_question": batch["layer2_questions"][i],
                    "stage1_options": o1,
                    "stage2_options": o2,
                    "stage1_generated_text": outputs["layer1_text"][i],
                    "stage2_generated_text": outputs["layer2_text"][i],
                }
                records.append(rec)
                if isinstance(source_values, str):
                    src = source_values
                else:
                    if len(source_values) != n:
                        raise ValueError(f"batch field 'feature_source': expected length {n}")
                    src = source_values[i]
                if src not in {"cache", "waveform"}:
                    raise ValueError(f"invalid feature_source for segment_id={segment_id!r}: {src!r}")
                if stream: stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
    metrics = meter.compute()
    if output_dir is not None:
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "predictions.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics, records
