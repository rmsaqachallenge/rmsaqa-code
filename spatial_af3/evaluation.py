from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tqdm import tqdm

from .mcq import extract_mcq_option
from .metrics import CategoryHierarchicalAccuracyMeter


def _reference_option(options: Dict[str, str], reference: str, stage: int) -> str:
    if not options:
        raise ValueError(f"Stage {stage} MCQ options are required for evaluation.")
    if not reference:
        raise ValueError(f"Stage {stage} correct_option is required for evaluation.")
    if reference not in options:
        raise ValueError(
            f"Stage {stage} correct_option {reference!r} is not present in options."
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
    return_counts: bool = False,
) -> tuple:
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
            for idx, segment_id in enumerate(batch["segment_ids"]):
                stage1_options = batch["stage1_options"][idx]
                stage2_options = batch["stage2_options"][idx]
                ref1 = batch["stage1_correct_option"][idx]
                ref2 = batch["stage2_correct_option"][idx]
                ref1_text = _reference_option(stage1_options, ref1, stage=1)
                ref2_text = _reference_option(stage2_options, ref2, stage=2)
                pred1, extract_status1 = extract_mcq_option(
                    outputs["stage1_text"][idx], stage1_options
                )
                pred2, extract_status2 = extract_mcq_option(
                    outputs["stage2_text"][idx], stage2_options
                )
                correct1 = pred1 == ref1
                correct2 = pred2 == ref2
                category_id = batch["stage2_category_ids"][idx]
                category_name = batch["stage2_category_names"][idx]
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
                    "stage2_category_id": category_id,
                    "stage2_category_name": category_name,
                    "stage1_pred_option": pred1,
                    "stage1_pred_option_text": (
                        stage1_options.get(pred1, "") if pred1 is not None else ""
                    ),
                    "stage1_extract_status": extract_status1,
                    "stage1_ref_option": ref1,
                    "stage1_ref_option_text": ref1_text,
                    "stage1_correct": correct1,
                    "stage2_pred_option": pred2,
                    "stage2_pred_option_text": (
                        stage2_options.get(pred2, "") if pred2 is not None else ""
                    ),
                    "stage2_extract_status": extract_status2,
                    "stage2_ref_option": ref2,
                    "stage2_ref_option_text": ref2_text,
                    "stage2_correct_raw": correct2,
                    "grounded_stage2_correct": bool(correct1 and correct2),
                    "stage1_question": batch["stage1_questions"][idx],
                    "stage2_question": batch["stage2_questions"][idx],
                    "stage1_options": stage1_options,
                    "stage2_options": stage2_options,
                    "stage1_generated_text": outputs["stage1_text"][idx],
                    "stage2_generated_text": outputs["stage2_text"][idx],
                    "stage1_gate": float(outputs["stage1_gate"][idx]),
                    "stage2_gate": float(outputs["stage2_gate"][idx]),
                }
                if collect_predictions:
                    pred_records.append(record)
                if log_f is not None:
                    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.set_postfix(
                grounded_stage2_acc=f"{meter.compute()['overall']['grounded_stage2_acc']:.4f}"
            )

    metrics = meter.compute()
    if return_counts:
        return metrics, pred_records, meter.counts()
    return metrics, pred_records
