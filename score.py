"""Calculate Stage-1, Stage-2, and joint accuracy from a results CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence


_DATA_FILE_PATTERN = re.compile(r"dev-(\d+)\.json\Z")
_CSV_FIELDS = ("Filename", "Stage-1 Answer", "Stage-2 Answer")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _normalise_option(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected an option letter at {location}, got {value!r}")
    option = value.strip().upper()
    if not re.fullmatch(r"[A-Z]", option):
        raise ValueError(f"Expected a single option letter at {location}, got {value!r}")
    return option


def _normalise_result_option(value: Any) -> str:
    """Normalize a prediction; blank or malformed predictions score as wrong."""

    if not isinstance(value, str):
        return ""
    option = value.strip().upper()
    return option if re.fullmatch(r"[A-Z]", option) else ""


def _normalise_filename(value: str) -> str:
    """Match a filename with or without its .json extension."""

    filename = value.strip()
    if filename.lower().endswith(".json"):
        filename = filename[:-5]
    return filename.casefold()


def _load_answer_record(path: Path) -> tuple[str, str, str]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Answer JSON must contain a records list: {path}")
    records = payload["records"]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"Expected exactly one answer record: {path}")

    record = records[0]
    segment_id = record.get("segment_id")
    qa_pairs = record.get("qa_pairs")
    if not isinstance(segment_id, str) or not segment_id.strip():
        raise ValueError(f"Invalid segment_id in answer file: {path}")
    if not isinstance(qa_pairs, list):
        raise ValueError(f"Answer record lacks qa_pairs: {path}")

    answers: dict[int, Any] = {}
    for pair in qa_pairs:
        if isinstance(pair, dict) and isinstance(pair.get("stage"), int):
            answers[pair["stage"]] = pair.get("answer")
    for stage in (1, 2):
        if stage not in answers:
            raise ValueError(f"Missing stage {stage} answer: {path}")
    return (
        segment_id,
        _normalise_option(answers[1], location=f"{path} stage 1"),
        _normalise_option(answers[2], location=f"{path} stage 2"),
    )


def _load_answers(data_dir: Path) -> dict[str, tuple[str, str]]:
    if not data_dir.is_dir():
        raise ValueError(f"Answer data directory not found: {data_dir}")

    indexed_paths: list[tuple[int, Path]] = []
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        match = _DATA_FILE_PATTERN.fullmatch(path.name)
        if match:
            indexed_paths.append((int(match.group(1)), path))
    indexed_paths.sort(key=lambda item: item[0])

    indices = [index for index, _ in indexed_paths]
    if indices != list(range(len(indices))):
        raise ValueError(
            "Answer files must be a contiguous numeric sequence starting at 0; "
            f"found indices {indices[:5]}...{indices[-5:] if indices else []}"
        )

    answers: dict[str, tuple[str, str]] = {}
    for _, path in indexed_paths:
        _, stage1, stage2 = _load_answer_record(path)
        filename_key = _normalise_filename(path.name)
        if filename_key in answers:
            raise ValueError(f"Duplicate answer filename: {path.name}")
        answers[filename_key] = (stage1, stage2)
    return answers


def _load_results(results_path: Path) -> list[dict[str, str]]:
    try:
        handle = results_path.open("r", newline="", encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ValueError(f"Results CSV not found: {results_path}") from exc

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(_CSV_FIELDS):
            raise ValueError(
                "Results CSV must have columns exactly "
                f"{list(_CSV_FIELDS)!r}; found {reader.fieldnames!r}"
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if any(row.get(field) is None for field in _CSV_FIELDS):
                raise ValueError(f"Missing CSV value at row {row_number}")
            filename = row["Filename"].strip()
            if not filename:
                raise ValueError(f"Empty Filename at CSV row {row_number}")
            rows.append(
                {
                    "Filename": filename,
                    "Stage-1 Answer": _normalise_result_option(row["Stage-1 Answer"]),
                    "Stage-2 Answer": _normalise_result_option(row["Stage-2 Answer"]),
                }
            )
    return rows


def calculate_metrics(results_path: str | Path, data_dir: str | Path) -> dict[str, float]:
    """Return Stage-1, Stage-2, and A_GS2 accuracy for a results CSV."""

    results = _load_results(Path(results_path))
    answers = _load_answers(Path(data_dir))
    if not results:
        raise ValueError("Results CSV contains no rows")

    stage1_correct = 0
    stage2_correct = 0
    joint_correct = 0
    seen_filenames: set[str] = set()
    for row in results:
        filename = row["Filename"]
        filename_key = _normalise_filename(filename)
        if filename_key in seen_filenames:
            raise ValueError(f"Duplicate Filename in results CSV: {filename}")
        seen_filenames.add(filename_key)
        if filename_key not in answers:
            raise ValueError(f"No answer found for Filename {filename!r}")

        answer_stage1, answer_stage2 = answers[filename_key]
        stage1_is_correct = row["Stage-1 Answer"] == answer_stage1
        stage2_is_correct = row["Stage-2 Answer"] == answer_stage2
        stage1_correct += stage1_is_correct
        stage2_correct += stage2_is_correct
        joint_correct += stage1_is_correct and stage2_is_correct

    missing_filenames = set(answers) - seen_filenames
    if missing_filenames:
        raise ValueError(
            "Results CSV does not cover all answer filenames; "
            f"missing {len(missing_filenames)}"
        )

    total = len(results)
    return {
        "stage1_accuracy": stage1_correct / total,
        "stage2_accuracy": stage2_correct / total,
        "a_gs2": joint_correct / total,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate Stage-1, Stage-2, and A_GS2 accuracy from a results CSV."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = calculate_metrics(args.results, args.data_dir)
    print(f"Stage-1 accuracy: {metrics['stage1_accuracy']:.6f}")
    print(f"Stage-2 accuracy: {metrics['stage2_accuracy']:.6f}")
    print(f"A_GS2: {metrics['a_gs2']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
