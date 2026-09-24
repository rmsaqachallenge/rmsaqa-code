from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


STAGE2_CATEGORIES = {
    1: "Sound Counting",
    2: "Source Localization",
    3: "Temporal Detection",
    4: "Spatial Relations",
    5: "Temporal Relations",
    6: "Action Prediction",
}


def validate_stage2_category(category_id: int, category_name: str) -> None:
    if type(category_id) is not int or category_id not in STAGE2_CATEGORIES:
        raise ValueError(f"Unsupported Stage 2 category_id: {category_id!r}")
    if category_name is None:
        raise ValueError("Stage 2 category_name is required.")
    expected_name = STAGE2_CATEGORIES[category_id]
    if category_name != expected_name:
        raise ValueError(
            f"Stage 2 category_name {category_name!r} does not match "
            f"category_id {category_id} ({expected_name!r})."
        )


@dataclass
class HierarchicalAccuracyMeter:
    num_records: int = 0
    stage1_correct: int = 0
    stage2_correct_raw: int = 0
    grounded_stage2_correct: int = 0

    def update(
        self,
        segment_id: str,
        stage1_correct: bool,
        stage2_correct: bool,
        stage1_pred: str,
        stage1_ref: str,
        stage2_pred: str,
        stage2_ref: str,
    ) -> None:
        self.num_records += 1
        self.stage1_correct += int(stage1_correct)
        self.stage2_correct_raw += int(stage2_correct)
        self.grounded_stage2_correct += int(stage1_correct and stage2_correct)

    def counts(self) -> Dict[str, int]:
        return {
            "num_records": self.num_records,
            "stage1_correct": self.stage1_correct,
            "stage2_correct_raw": self.stage2_correct_raw,
            "grounded_stage2_correct": self.grounded_stage2_correct,
        }

    def compute(self) -> Dict[str, float]:
        denom = max(self.num_records, 1)
        return {
            "stage1_acc": self.stage1_correct / denom,
            "stage2_acc_raw": self.stage2_correct_raw / denom,
            "grounded_stage2_acc": self.grounded_stage2_correct / denom,
        }


class CategoryHierarchicalAccuracyMeter:
    def __init__(self) -> None:
        self.overall = HierarchicalAccuracyMeter()
        self.categories = {
            category_id: HierarchicalAccuracyMeter()
            for category_id in STAGE2_CATEGORIES
        }

    @staticmethod
    def _validate_category(category_id: int, category_name: str) -> None:
        validate_stage2_category(category_id, category_name)

    def update(
        self,
        *,
        category_id: int,
        category_name: str,
        segment_id: str,
        stage1_correct: bool,
        stage2_correct: bool,
        stage1_pred: str,
        stage1_ref: str,
        stage2_pred: str,
        stage2_ref: str,
    ) -> None:
        self._validate_category(category_id, category_name)
        update_args = {
            "segment_id": segment_id,
            "stage1_correct": stage1_correct,
            "stage2_correct": stage2_correct,
            "stage1_pred": stage1_pred,
            "stage1_ref": stage1_ref,
            "stage2_pred": stage2_pred,
            "stage2_ref": stage2_ref,
        }
        self.overall.update(**update_args)
        self.categories[category_id].update(**update_args)

    def counts(self) -> Dict[str, object]:
        category_counts = {}
        for category_id, category_name in STAGE2_CATEGORIES.items():
            category_counts[str(category_id)] = {
                "category_id": category_id,
                "category_name": category_name,
                **self.categories[category_id].counts(),
            }
        return {
            "overall": self.overall.counts(),
            "categories": category_counts,
        }

    def compute(self) -> Dict[str, object]:
        category_metrics = {}
        for category_id, category_name in STAGE2_CATEGORIES.items():
            category_metrics[str(category_id)] = {
                "category_id": category_id,
                "category_name": category_name,
                **self.categories[category_id].compute(),
            }
        return {
            "overall": self.overall.compute(),
            "categories": category_metrics,
        }
