from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch
from tqdm import tqdm

from .layered_mcq import extract_mcq_option
from .metrics import CategoryHierarchicalAccuracyMeter, validate_stage2_category


BATCH_REQUIRED_FIELDS = (
    "segment_ids",
    "audio_paths",
    "layer1_questions",
    "layer2_questions",
    "layer1_options",
    "layer2_options",
    "layer1_correct_option",
    "layer2_correct_option",
    "layer2_category_ids",
    "layer2_category_names",
)
OUTPUT_REQUIRED_FIELDS = (
    "layer1_text",
    "layer2_text",
    "layer1_gate",
    "layer2_gate",
)


def _require_fields(container: Mapping, container_name: str, fields) -> None:
    for field in fields:
        if field not in container:
            raise ValueError(
                f"{container_name} field {field!r}: expected present; "
                "actual missing."
            )


def _sequence_length(value, container_name: str, field: str) -> int:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not hasattr(
        value, "__getitem__"
    ):
        raise TypeError(
            f"{container_name} field {field!r}: expected an indexable sequence; "
            f"actual {type(value).__name__}."
        )
    try:
        return len(value)
    except TypeError as exc:
        raise TypeError(
            f"{container_name} field {field!r}: expected an indexable sequence; "
            f"actual {type(value).__name__}."
        ) from exc


def validate_generated_batch(batch, outputs) -> None:
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"outputs must be a Mapping; actual {type(outputs).__name__}."
        )
    if not isinstance(batch, Mapping):
        raise TypeError(f"batch must be a Mapping; actual {type(batch).__name__}.")

    _require_fields(batch, "batch", BATCH_REQUIRED_FIELDS)
    _require_fields(outputs, "outputs", OUTPUT_REQUIRED_FIELDS)
    expected = _sequence_length(batch["segment_ids"], "batch", "segment_ids")
    for field in BATCH_REQUIRED_FIELDS:
        actual = _sequence_length(batch[field], "batch", field)
        if actual != expected:
            raise ValueError(
                f"batch field {field!r}: expected length {expected}; "
                f"actual length {actual}."
            )
    if "task_ids" in batch:
        actual = _sequence_length(batch["task_ids"], "batch", "task_ids")
        if actual != expected:
            raise ValueError(
                f"batch field 'task_ids': expected length {expected}; "
                f"actual length {actual}."
            )
    for field in OUTPUT_REQUIRED_FIELDS:
        actual = _sequence_length(outputs[field], "outputs", field)
        if actual != expected:
            raise ValueError(
                f"outputs field {field!r}: expected length {expected}; "
                f"actual length {actual}."
            )
    for category_id, category_name in zip(
        batch["layer2_category_ids"],
        batch["layer2_category_names"],
    ):
        validate_stage2_category(category_id, category_name)


def _reference_option(options: Dict[str, str], reference: str, layer: int) -> str:
    if not options:
        raise ValueError(f"Layer {layer} MCQ options are required for evaluation.")
    if not reference:
        raise ValueError(f"Layer {layer} correct_option is required for evaluation.")
    if reference not in options:
        raise ValueError(
            f"Layer {layer} correct_option {reference!r} is not present in options."
        )
    return options[reference]


@torch.inference_mode()
def evaluate_spatial_model(
    model,
    loader: Iterable[Dict],
    device: torch.device,
    *,
    max_new_tokens: int,
    num_beams: int,
    desc: str,
    collect_predictions: bool = True,
    answers_log_path: Optional[Path] = None,
    show_progress: bool = True,
    validate_batch: Optional[Callable[[Mapping, Mapping], None]] = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    meter = CategoryHierarchicalAccuracyMeter()
    pred_records: List[Dict[str, object]] = []

    if answers_log_path is not None:
        answers_log_path = Path(answers_log_path)
        answers_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_context = answers_log_path.open("w", encoding="utf-8")
    else:
        log_context = nullcontext(None)

    with log_context as log_f:
        pbar = tqdm(loader, desc=desc, disable=not show_progress)
        for batch in pbar:
            outputs = model.generate_freeform_answers(
                batch,
                device=device,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            if validate_batch is not None:
                validate_batch(batch, outputs)
            for idx, segment_id in enumerate(batch["segment_ids"]):
                layer1_options = batch["layer1_options"][idx]
                layer2_options = batch["layer2_options"][idx]
                ref1 = batch["layer1_correct_option"][idx]
                ref2 = batch["layer2_correct_option"][idx]
                ref1_text = _reference_option(layer1_options, ref1, layer=1)
                ref2_text = _reference_option(layer2_options, ref2, layer=2)
                pred1, extract_status1 = extract_mcq_option(
                    outputs["layer1_text"][idx], layer1_options
                )
                pred2, extract_status2 = extract_mcq_option(
                    outputs["layer2_text"][idx], layer2_options
                )
                correct1 = pred1 == ref1
                correct2 = pred2 == ref2
                category_id = batch["layer2_category_ids"][idx]
                category_name = batch["layer2_category_names"][idx]
                meter.update(
                    category_id=category_id,
                    category_name=category_name,
                    segment_id=segment_id,
                    stage1_correct=correct1,
                    stage2_correct=correct2,
                    stage1_pred=pred1,
                    stage1_ref=ref1,
                    stage2_pred=pred2,
                    stage2_ref=ref2,
                )
                task_ids = batch.get("task_ids")
                task_id = task_ids[idx] if task_ids is not None else None
                record = {
                    "segment_id": segment_id,
                    "audio_path": batch["audio_paths"][idx],
                    "task_id": task_id,
                    "layer2_category_id": category_id,
                    "layer2_category_name": category_name,
                    "layer1_pred_option": pred1,
                    "layer1_pred_option_text": (
                        layer1_options.get(pred1, "") if pred1 is not None else ""
                    ),
                    "layer1_extract_status": extract_status1,
                    "layer1_ref_option": ref1,
                    "layer1_ref_option_text": ref1_text,
                    "layer1_correct": correct1,
                    "layer2_pred_option": pred2,
                    "layer2_pred_option_text": (
                        layer2_options.get(pred2, "") if pred2 is not None else ""
                    ),
                    "layer2_extract_status": extract_status2,
                    "layer2_ref_option": ref2,
                    "layer2_ref_option_text": ref2_text,
                    "layer2_correct_raw": correct2,
                    "layer2_correct_hier": bool(correct1 and correct2),
                    "layer1_question": batch["layer1_questions"][idx],
                    "layer2_question": batch["layer2_questions"][idx],
                    "layer1_options": layer1_options,
                    "layer2_options": layer2_options,
                    "layer1_generated_text": outputs["layer1_text"][idx],
                    "layer2_generated_text": outputs["layer2_text"][idx],
                    "layer1_gate": float(outputs["layer1_gate"][idx]),
                    "layer2_gate": float(outputs["layer2_gate"][idx]),
                }
                if collect_predictions:
                    pred_records.append(record)
                if log_f is not None:
                    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.set_postfix(
                grounded_stage2_acc=f"{meter.compute()['overall']['grounded_stage2_acc']:.4f}"
            )

    metrics = meter.compute()
    counts = meter.counts()
    metrics["overall"]["num_records"] = counts["overall"]["num_records"]
    for category_key, category_counts in counts["categories"].items():
        metrics["categories"][category_key]["num_records"] = category_counts[
            "num_records"
        ]
    return metrics, pred_records
