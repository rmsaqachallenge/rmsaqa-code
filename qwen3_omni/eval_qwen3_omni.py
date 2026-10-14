#!/usr/bin/env python
"""Single-device Qwen3-Omni evaluation entry point.

Startup validation, dataset preflight, lazy model construction, strict
checkpoint loading, and atomic evaluation artifact publication are implemented
here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from torch.utils.data import DataLoader, SequentialSampler

from spatial_af3.layered_dataset import SpatialQAPairDataset, qa_pair_collate_fn
from spatial_af3.layered_evaluation import evaluate_spatial_model, validate_generated_batch
from spatial_af3.metrics import validate_stage2_category


FIXED_OUTPUT_NAMES = (
    "eval_args.json",
    "metrics.json",
    "predictions.json",
)
LORA_ARG_NAMES = (
    "llm_lora_r",
    "llm_lora_alpha",
    "llm_lora_dropout",
    "llm_lora_num_layers",
)
EVALUATION_ARG_NAMES = (
    "qwen3_omni_root",
    "test_json",
    "feature_root",
    "checkpoint",
    "output_dir",
    "batch_size",
    "num_workers",
    "max_records",
    "extract_features_on_the_fly",
    "write_missing_features",
    "audio_token_stride",
    "max_new_tokens",
    "num_beams",
    "answers_log_name",
    "device",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a labeled FOA split with Qwen3-Omni on one device."
    )
    parser.add_argument("--qwen3-omni-root", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--extract-features-on-the-fly", action="store_true")
    parser.add_argument("--write-missing-features", action="store_true")
    parser.add_argument("--audio-token-stride", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--answers-log-name", default="answers_log.jsonl")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def validate_answers_log_name(name: str) -> None:
    raw_name = str(name)
    raw_terminal_part = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
    posix_path = PurePosixPath(raw_name.replace("\\", "/"))
    windows_path = PureWindowsPath(raw_name)
    has_reserved_part = any(
        PureWindowsPath(part).is_reserved() for part in windows_path.parts
    )
    has_invalid_part = any(
        part.endswith((".", " "))
        or any(
            character in '<>:"|?*' or unicodedata.category(character) == "Cc"
            for character in part
        )
        for part in windows_path.parts
    )
    collides_with_fixed_output = len(
        posix_path.parts
    ) == 1 and posix_path.name.casefold() in {
        name.casefold() for name in FIXED_OUTPUT_NAMES
    }
    if (
        not raw_name.strip()
        or raw_terminal_part in {"", "."}
        or Path(raw_name).is_absolute()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
        or has_reserved_part
        or has_invalid_part
        or not windows_path.name
    ):
        raise ValueError(
            "--answers-log-name must be a safe, non-empty relative file path "
            "without '..'."
        )
    if collides_with_fixed_output:
        raise ValueError(
            "--answers-log-name collides with a fixed output artifact: "
            f"{posix_path.name}."
        )


def resolve_answers_log_path(output_dir: str | Path, answers_log_name: str) -> Path:
    validate_answers_log_name(answers_log_name)
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_log_path = (resolved_output_dir / answers_log_name).resolve()
    if resolved_output_dir not in resolved_log_path.parents:
        raise ValueError(
            "--answers-log-name must resolve to a file within --output-dir."
        )
    for fixed_name in FIXED_OUTPUT_NAMES:
        fixed_path = (resolved_output_dir / fixed_name).resolve()
        if str(resolved_log_path).casefold() == str(fixed_path).casefold():
            raise ValueError(
                "--answers-log-name collides with a fixed output artifact: "
                f"{fixed_name}."
            )
    return resolved_log_path


def validate_args(args: argparse.Namespace) -> None:
    for name in ("extract_features_on_the_fly", "write_missing_features"):
        if type(getattr(args, name)) is not bool:
            raise ValueError(f"--{name.replace('_', '-')} must be a boolean.")

    integer_values = {
        "batch-size": args.batch_size,
        "num-workers": args.num_workers,
        "max-records": args.max_records,
        "audio-token-stride": args.audio_token_stride,
        "max-new-tokens": args.max_new_tokens,
        "num-beams": args.num_beams,
    }
    for name, value in integer_values.items():
        if name == "max-records" and value is None:
            continue
        if type(value) is not int:
            raise ValueError(f"--{name} must be an integer.")

    for name in (
        "qwen3_omni_root",
        "test_json",
        "feature_root",
        "checkpoint",
        "output_dir",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"--{name.replace('_', '-')} must be a non-empty path string."
            )

    for name, value in {
        "batch-size": args.batch_size,
        "max-new-tokens": args.max_new_tokens,
        "num-beams": args.num_beams,
    }.items():
        if value <= 0:
            raise ValueError(f"--{name} must be greater than zero.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be greater than zero when provided.")
    if not 1 <= args.audio_token_stride <= 1500:
        raise ValueError("--audio-token-stride must be between 1 and 1500.")
    if args.write_missing_features and not args.extract_features_on_the_fly:
        raise ValueError(
            "--write-missing-features requires --extract-features-on-the-fly."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_args(args)
    validate_answers_log_name(args.answers_log_name)
    return args


def normalize_runtime_paths(args: argparse.Namespace) -> argparse.Namespace:
    directory_fields = ("qwen3_omni_root", "feature_root")
    file_fields = ("checkpoint",)
    for field in directory_fields:
        path = Path(getattr(args, field)).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(
                f"--{field.replace('_', '-')} must be an existing directory: {path}"
            )
        setattr(args, field, str(path))
    test_json = Path(args.test_json).expanduser().resolve()
    if not test_json.is_dir() and not (
        test_json.is_file() and test_json.suffix.lower() == ".json"
    ):
        raise ValueError(
            "--test-json must be an existing JSON file or directory: "
            f"{test_json}"
        )
    args.test_json = str(test_json)
    for field in file_fields:
        path = Path(getattr(args, field)).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"--{field.replace('_', '-')} must be an existing file: {path}"
            )
        setattr(args, field, str(path))

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir must be a directory path: {output_dir}")
    args.output_dir = str(output_dir)
    return args


def load_checkpoint_payload(
    checkpoint_path: str | Path,
) -> tuple[Mapping[str, Any], dict[str, int | float]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint state_dict must be a mapping.")
    if any(not isinstance(key, str) for key in state_dict):
        raise ValueError("Checkpoint state_dict keys must be strings.")
    checkpoint_args = payload.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Checkpoint args must be a mapping.")

    config: dict[str, int | float] = {}
    for name in ("llm_lora_r", "llm_lora_alpha", "llm_lora_num_layers"):
        value = checkpoint_args.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Checkpoint {name} must be an exact positive integer.")
        config[name] = value

    dropout = checkpoint_args.get("llm_lora_dropout")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, Real)
        or not 0 <= dropout < 1
        or not math.isfinite(dropout)
    ):
        raise ValueError(
            "Checkpoint llm_lora_dropout must be a finite real value in [0, 1)."
        )
    config["llm_lora_dropout"] = float(dropout)
    return payload, {name: config[name] for name in LORA_ARG_NAMES}


def _get_model_constructor():
    from spatial_af3.models import SpatialQwen3OmniGenerativeModel

    return SpatialQwen3OmniGenerativeModel


def build_model(args: argparse.Namespace, lora_config: Mapping[str, object]):
    return _get_model_constructor()(
        qwen3_omni_root=args.qwen3_omni_root,
        text_dtype=torch.bfloat16,
        audio_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        enable_llm_lora=True,
        llm_lora_r=lora_config["llm_lora_r"],
        llm_lora_alpha=lora_config["llm_lora_alpha"],
        llm_lora_dropout=lora_config["llm_lora_dropout"],
        llm_lora_num_layers=lora_config["llm_lora_num_layers"],
    )


def required_checkpoint_names(model: torch.nn.Module) -> set[str]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    buffer_names: set[str] = set()
    for module_name, module in model.named_modules():
        has_direct_trainable_parameter = any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        )
        if not has_direct_trainable_parameter:
            continue
        for buffer_name, _buffer in module.named_buffers(recurse=False):
            if buffer_name in module._non_persistent_buffers_set:
                continue
            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            buffer_names.add(full_name)

    required_names = trainable_names | buffer_names
    model_state = model.state_dict()
    if not isinstance(model_state, Mapping):
        raise TypeError("Model state_dict must be a mapping.")
    missing_names = required_names.difference(model_state)
    if missing_names:
        missing_text = ", ".join(sorted(missing_names))
        raise ValueError(
            f"Required checkpoint name(s) {missing_text} are missing from model state_dict."
        )
    return required_names


def _validate_checkpoint_tensor(value: object, *, name: object, owner: str) -> None:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{owner} '{name}' must be an exact torch.Tensor.")
    if value.device.type == "meta":
        raise ValueError(f"{owner} '{name}' must contain materialized values.")
    if value.layout != torch.strided:
        raise ValueError(f"{owner} '{name}' must use strided layout.")
    if value.is_quantized:
        raise ValueError(f"{owner} '{name}' must be non-quantized.")


def load_required_checkpoint_state(
    model: torch.nn.Module, payload: Mapping[str, object]
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping.")

    checkpoint_items = list(state_dict.items())
    if any(not isinstance(name, str) for name, _value in checkpoint_items):
        raise TypeError("Checkpoint state_dict keys must be strings.")
    for name, value in checkpoint_items:
        _validate_checkpoint_tensor(value, name=name, owner="Checkpoint state")

    required_set = required_checkpoint_names(model)
    supplied_set = {name for name, _value in checkpoint_items}
    missing_names = required_set - supplied_set
    extra_names = supplied_set - required_set
    if missing_names or extra_names:
        details = []
        if missing_names:
            details.append("missing: " + ", ".join(sorted(missing_names)))
        if extra_names:
            details.append("unexpected: " + ", ".join(sorted(extra_names)))
        raise ValueError(
            "Checkpoint state names must exactly match required names ("
            + "; ".join(details)
            + ")."
        )

    target_state = model.state_dict()
    if not isinstance(target_state, Mapping):
        raise TypeError("Model state_dict must be a mapping.")
    for name in sorted(required_set):
        target = target_state[name]
        _validate_checkpoint_tensor(target, name=name, owner="Model state")
        checkpoint_value = state_dict[name]
        if checkpoint_value.shape != target.shape:
            raise ValueError(
                f"Checkpoint state '{name}' shape {tuple(checkpoint_value.shape)} "
                f"does not match model shape {tuple(target.shape)}."
            )

    target_references = {name: target_state[name] for name in required_set}
    original_values = {
        name: target.detach().clone() for name, target in target_references.items()
    }
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
        remaining_required = required_set.intersection(incompatible.missing_keys)
        if remaining_required:
            raise ValueError(
                "Required checkpoint state remained missing after load: "
                + ", ".join(sorted(remaining_required))
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                "Unexpected checkpoint state after load: "
                + ", ".join(sorted(incompatible.unexpected_keys))
            )
    except BaseException as error:
        restoration_errors = []
        with torch.no_grad():
            for name in sorted(required_set):
                try:
                    target_references[name].copy_(original_values[name])
                except BaseException as restoration_error:
                    restoration_errors.append(f"{name}: {restoration_error!r}")
        if restoration_errors:
            note = "Checkpoint rollback also failed for " + "; ".join(
                restoration_errors
            )
            add_note = getattr(error, "add_note", None)
            if add_note is not None:
                add_note(note)
            else:
                error.args = (*error.args, note)
        raise


def _get_validate_qwen3_dataset_sample():
    from train_qwen3_omni import validate_qwen3_dataset_sample

    return validate_qwen3_dataset_sample


def validate_evaluation_sample(sample: Mapping[str, object]) -> None:
    _get_validate_qwen3_dataset_sample()(sample, "evaluation")
    validate_stage2_category(
        sample["layer2_category_id"], sample["layer2_category_name"]
    )


def build_dataset(args: argparse.Namespace) -> SpatialQAPairDataset:
    dataset = SpatialQAPairDataset(
        json_path=args.test_json,
        feature_root=args.feature_root,
        extract_features_on_the_fly=args.extract_features_on_the_fly,
        write_missing_features=args.write_missing_features,
        audio_token_stride=args.audio_token_stride,
        max_records=args.max_records,
    )
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")
    validate_evaluation_sample(dataset[0])
    return dataset


def build_loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=qa_pair_collate_fn,
        pin_memory=True,
    )


def _add_cleanup_note(error: BaseException, message: str) -> None:
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(message)
    else:
        error.args = (*error.args, message)


def _unlink_staging(path: Path, error: BaseException | None = None) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except BaseException as cleanup_error:
        if error is None:
            raise
        _add_cleanup_note(error, f"Cleanup also failed for '{path}': {cleanup_error!r}")


def _stage_json(value: object, destination: Path) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.stage.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.close(file_descriptor)
    except BaseException as error:
        _unlink_staging(temporary_path, error)
        raise
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
    except BaseException as error:
        _unlink_staging(temporary_path, error)
        raise
    return temporary_path


def atomic_dump_json(value: object, destination: str | Path) -> None:
    destination_path = Path(destination)
    temporary_path = _stage_json(value, destination_path)
    try:
        os.replace(temporary_path, destination_path)
    except BaseException as error:
        _unlink_staging(temporary_path, error)
        raise


def _backup_file(destination: Path) -> Path:
    file_descriptor, backup_name = tempfile.mkstemp(
        prefix=f".{destination.name}.backup.",
        suffix=".tmp",
        dir=destination.parent,
    )
    backup_path = Path(backup_name)
    try:
        os.close(file_descriptor)
    except BaseException as error:
        _unlink_staging(backup_path, error)
        raise
    try:
        shutil.copyfile(destination, backup_path)
    except BaseException as error:
        _unlink_staging(backup_path, error)
        raise
    return backup_path


def atomic_dump_json_pair(
    first_value: object,
    first_destination: str | Path,
    second_value: object,
    second_destination: str | Path,
) -> None:
    destinations = (Path(first_destination), Path(second_destination))
    stages: list[Path] = []
    backups: list[Path | None] = []
    existed = tuple(destination.exists() for destination in destinations)
    try:
        stages.append(_stage_json(first_value, destinations[0]))
        stages.append(_stage_json(second_value, destinations[1]))
        for destination, was_present in zip(destinations, existed):
            backups.append(_backup_file(destination) if was_present else None)
    except BaseException as error:
        for path in (*stages, *(path for path in backups if path is not None)):
            _unlink_staging(path, error)
        raise

    try:
        for stage, destination in zip(stages, destinations):
            os.replace(stage, destination)
    except BaseException as error:
        for index, (destination, backup, was_present) in enumerate(
            zip(destinations, backups, existed)
        ):
            try:
                if was_present:
                    assert backup is not None
                    os.replace(backup, destination)
                    backups[index] = None
                else:
                    destination.unlink(missing_ok=True)
            except BaseException as rollback_error:
                if backup is not None and backup.exists():
                    _add_cleanup_note(
                        error,
                        f"Rollback also failed for '{destination}': "
                        f"{rollback_error!r}. Recovery backup retained at "
                        f"'{backup}'.",
                    )
                else:
                    _add_cleanup_note(
                        error,
                        f"Rollback also failed for '{destination}': "
                        f"{rollback_error!r}.",
                    )
        for path in stages:
            _unlink_staging(path, error)
        raise

    for backup in backups:
        if backup is not None:
            _unlink_staging(backup)


def checkpoint_runtime_metadata(payload: Mapping[str, object]) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping.")
    metadata: dict[str, int] = {}
    for name in ("epoch", "step"):
        if name not in payload:
            continue
        value = payload[name]
        if type(value) is not int or value < 0:
            raise ValueError(
                f"Checkpoint {name} must be an exact non-negative integer when present."
            )
        metadata[name] = value
    return metadata


def evaluation_args_record(
    args: argparse.Namespace,
    lora_config: Mapping[str, object],
    payload: Mapping[str, object],
) -> dict[str, object]:
    record = {name: getattr(args, name) for name in EVALUATION_ARG_NAMES}
    record.update({name: lora_config[name] for name in LORA_ARG_NAMES})
    metadata = checkpoint_runtime_metadata(payload)
    if "epoch" in metadata:
        record["checkpoint_epoch"] = metadata["epoch"]
    if "step" in metadata:
        record["checkpoint_step"] = metadata["step"]
    return record


def resolve_device(device_name: object) -> torch.device:
    try:
        device = torch.device(device_name)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"--device is invalid: {device_name!r}.") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device supports only CPU and CUDA devices.")
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU --device must not include a device index.")
        return device
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested through --device but is unavailable.")
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise ValueError("CUDA was requested through --device but is unavailable.")
    if device.index is not None and device.index >= device_count:
        raise ValueError(
            f"CUDA device index {device.index} is out of range for "
            f"{device_count} available device(s)."
        )
    return device


def prepare_output_directory(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_descriptor, probe_name = tempfile.mkstemp(
        prefix=".eval-write-probe.", suffix=".tmp", dir=output_path
    )
    probe_path = Path(probe_name)
    try:
        os.close(file_descriptor)
    except BaseException as error:
        _unlink_staging(probe_path, error)
        raise
    _unlink_staging(probe_path)
    return output_path


def prepare_answers_log_path(answers_log_path: str | Path) -> Path:
    path = Path(answers_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, probe_name = tempfile.mkstemp(
        prefix=f".{path.name}.write-probe.", suffix=".tmp", dir=path.parent
    )
    probe_path = Path(probe_name)
    try:
        os.close(file_descriptor)
    except BaseException as error:
        _unlink_staging(probe_path, error)
        raise
    _unlink_staging(probe_path)
    return path


def validate_artifact_destinations(
    output_dir: str | Path, answers_log_path: str | Path
) -> None:
    output_path = Path(output_dir)
    answers_path = Path(answers_log_path)
    destinations = (
        answers_path,
        *(output_path / name for name in FIXED_OUTPUT_NAMES),
    )
    for destination in destinations:
        if destination.exists() and not stat.S_ISREG(destination.stat().st_mode):
            raise ValueError(
                f"Artifact destination '{destination}' must be a regular file."
            )
    if answers_path.exists():
        with answers_path.open("ab"):
            pass


def evaluate(
    model,
    loader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str | Path,
    answers_log_path: str | Path,
):
    output_path = Path(output_dir)
    answers_log_path = Path(answers_log_path)
    metrics, predictions = evaluate_spatial_model(
        model,
        loader,
        device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        desc="eval",
        answers_log_path=answers_log_path,
        collect_predictions=True,
        show_progress=True,
        validate_batch=validate_generated_batch,
    )
    atomic_dump_json_pair(
        metrics,
        output_path / "metrics.json",
        predictions,
        output_path / "predictions.json",
    )
    return metrics, predictions


def run_evaluation(args: argparse.Namespace, *, evaluator=None, layer1_context: str | None = None):
    if layer1_context is not None:
        if layer1_context not in {"none", "gold"} or evaluator is None:
            raise ValueError("Layer2 context must be 'none' or 'gold' with a dedicated evaluator.")
    validate_args(args)
    validate_answers_log_name(args.answers_log_name)
    normalize_runtime_paths(args)
    payload, lora_config = load_checkpoint_payload(args.checkpoint)
    checkpoint_runtime_metadata(payload)
    answers_log_path = resolve_answers_log_path(args.output_dir, args.answers_log_name)
    dataset = build_dataset(args)
    loader = build_loader(dataset, args)
    device = resolve_device(args.device)
    output_dir = prepare_output_directory(args.output_dir)
    prepare_answers_log_path(answers_log_path)
    validate_artifact_destinations(output_dir, answers_log_path)
    args_record = evaluation_args_record(args, lora_config, payload)
    if layer1_context is not None:
        args_record["layer1_context"] = layer1_context
    json.dumps(args_record, ensure_ascii=False, indent=2, allow_nan=False)

    model = build_model(args, lora_config)
    load_required_checkpoint_state(model, payload)
    del payload
    model.to(device)
    model.eval()

    atomic_dump_json(args_record, output_dir / "eval_args.json")
    evaluate_fn = evaluate if evaluator is None else evaluator
    return evaluate_fn(model, loader, args, device, output_dir, answers_log_path)


def main(argv: Sequence[str] | None = None):
    return run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
