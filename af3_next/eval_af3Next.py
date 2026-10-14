#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from spatial_af3.layered_dataset import SpatialQAPairDataset, qa_pair_collate_fn
from spatial_af3.layered_evaluation import evaluate_spatial_model, validate_generated_batch
from spatial_af3.models import SpatialAFNextGenerativeModel
from spatial_af3.utils import dump_json, ensure_dir


LORA_ARG_NAMES = (
    "llm_lora_r",
    "llm_lora_alpha",
    "llm_lora_dropout",
    "llm_lora_num_layers",
)
FIXED_OUTPUT_NAMES = (
    "eval_args.json",
    "metrics.json",
    "predictions.json",
)


def ensure_foa_json(json_path: str | Path, arg_name: str) -> Path:
    del arg_name
    return Path(json_path).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spatial AF3-Next concat-fusion model."
    )
    parser.add_argument("--afnext-root", required=True)
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
    parser.add_argument("--llm-lora-r", type=int, default=16)
    parser.add_argument("--llm-lora-alpha", type=int, default=16)
    parser.add_argument("--llm-lora-dropout", type=float, default=0.05)
    parser.add_argument("--llm-lora-num-layers", type=int, default=28)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--answers-log-name", default="answers_log.jsonl")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--audio-encoder-device", default="cpu")
    return parser


def validate_answers_log_name(name: str) -> None:
    raw_name = str(name)
    posix_path = PurePosixPath(raw_name.replace("\\", "/"))
    windows_path = PureWindowsPath(raw_name)
    has_reserved_windows_part = any(
        PureWindowsPath(part).is_reserved() for part in windows_path.parts
    )
    has_windows_ads_colon = any(":" in part for part in windows_path.parts)
    has_invalid_windows_component = any(
        part.endswith((".", " "))
        or any(
            character in '<>:"|?*'
            or unicodedata.category(character) == "Cc"
            for character in part
        )
        for part in windows_path.parts
    )
    collides_with_fixed_output = (
        len(posix_path.parts) == 1
        and posix_path.name.casefold()
        in {fixed_name.casefold() for fixed_name in FIXED_OUTPUT_NAMES}
    )
    if (
        not raw_name.strip()
        or Path(raw_name).is_absolute()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
        or has_reserved_windows_part
        or has_windows_ads_colon
        or has_invalid_windows_component
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


def resolve_answers_log_path(
    output_dir: str | Path, answers_log_name: str
) -> Path:
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
    positive_values = {
        "batch-size": args.batch_size,
        "llm-lora-r": args.llm_lora_r,
        "llm-lora-alpha": args.llm_lora_alpha,
        "llm-lora-num-layers": args.llm_lora_num_layers,
        "max-new-tokens": args.max_new_tokens,
        "num-beams": args.num_beams,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} must be greater than zero.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if not 1 <= args.audio_token_stride <= 1500:
        raise ValueError("--audio-token-stride must be between 1 and 1500.")
    if not math.isfinite(args.llm_lora_dropout) or not (
        0 <= args.llm_lora_dropout < 1
    ):
        raise ValueError("--llm-lora-dropout must be finite and in [0, 1).")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be greater than zero when provided.")
    if args.write_missing_features and not args.extract_features_on_the_fly:
        raise ValueError(
            "--write-missing-features requires --extract-features-on-the-fly."
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_args(args)
    validate_answers_log_name(args.answers_log_name)
    return args


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
    return dataset


def build_loader(
    dataset: SpatialQAPairDataset, args: argparse.Namespace
) -> DataLoader:
    sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=qa_pair_collate_fn,
        pin_memory=True,
    )


def build_model(args: argparse.Namespace) -> SpatialAFNextGenerativeModel:
    return SpatialAFNextGenerativeModel(
        afnext_root=args.afnext_root,
        enable_llm_lora=True,
        llm_lora_r=args.llm_lora_r,
        llm_lora_alpha=args.llm_lora_alpha,
        llm_lora_dropout=args.llm_lora_dropout,
        llm_lora_num_layers=args.llm_lora_num_layers,
    )


def required_checkpoint_names(model: nn.Module) -> set[str]:
    model_state = model.state_dict()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_trainable_names = trainable_names - set(model_state)
    if missing_trainable_names:
        raise ValueError(
            "Model trainable parameters are missing from state_dict: "
            + ", ".join(sorted(missing_trainable_names))
        )
    related_buffer_names = set()
    for module_name, module in model.named_modules():
        if any(
            parameter.requires_grad
            for parameter in module.parameters(recurse=False)
        ):
            for buffer_name, _ in module.named_buffers(recurse=False):
                qualified_name = (
                    f"{module_name}.{buffer_name}" if module_name else buffer_name
                )
                if qualified_name in model_state:
                    related_buffer_names.add(qualified_name)
    return trainable_names | related_buffer_names


def load_required_checkpoint(
    model: nn.Module, checkpoint_path: str | Path, args: argparse.Namespace
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a Mapping.")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a Mapping.")
    non_string_keys = [key for key in state_dict if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(
            "Checkpoint state_dict keys must be strings; invalid keys: "
            + ", ".join(repr(key) for key in non_string_keys)
        )

    checkpoint_args = payload.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise TypeError("Checkpoint args must be a Mapping.")

    for name in LORA_ARG_NAMES:
        if name not in checkpoint_args:
            raise ValueError(f"Checkpoint args missing LoRA field: {name}")
        checkpoint_value = checkpoint_args[name]
        if name == "llm_lora_dropout":
            if (
                not isinstance(checkpoint_value, Real)
                or isinstance(checkpoint_value, bool)
                or not math.isfinite(checkpoint_value)
            ):
                raise ValueError(
                    f"Checkpoint LoRA field {name} must be a finite Real scalar."
                )
        elif type(checkpoint_value) is not int:
            raise ValueError(
                f"Checkpoint LoRA field {name} must be a Python int."
            )
        current_value = getattr(args, name)
        if checkpoint_value != current_value:
            raise ValueError(
                f"Checkpoint LoRA field {name} does not match current args: "
                f"{checkpoint_value!r} != {current_value!r}"
            )

    required_names = required_checkpoint_names(model)
    checkpoint_names = set(state_dict)
    missing_required = required_names - checkpoint_names
    if missing_required:
        raise ValueError(
            "Checkpoint is missing required state: "
            + ", ".join(sorted(missing_required))
        )

    model_state = model.state_dict()
    unexpected_names = checkpoint_names - set(model_state)
    if unexpected_names:
        raise ValueError(
            "Checkpoint contains unexpected state: "
            + ", ".join(sorted(unexpected_names))
        )

    incompatible_values = []
    for name, value in state_dict.items():
        expected_value = model_state[name]
        if isinstance(expected_value, torch.Tensor):
            expected = f"Tensor shape {tuple(expected_value.shape)}"
            if not isinstance(value, torch.Tensor):
                actual = f"{type(value).__name__} (non-Tensor)"
                incompatible_values.append(f"{name}: expected {expected}, actual {actual}")
            elif value.shape != expected_value.shape:
                actual = f"Tensor shape {tuple(value.shape)}"
                incompatible_values.append(f"{name}: expected {expected}, actual {actual}")
    if incompatible_values:
        raise ValueError(
            "Checkpoint contains incompatible state values: "
            + "; ".join(incompatible_values)
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    remaining_required = required_names.intersection(incompatible.missing_keys)
    if remaining_required:
        raise ValueError(
            "Checkpoint is missing required state after loading: "
            + ", ".join(sorted(remaining_required))
        )
    if incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint contains unexpected state after loading: "
            + ", ".join(sorted(incompatible.unexpected_keys))
        )


def _validate_generated_batch(batch, outputs) -> None:
    validate_generated_batch(batch, outputs)


def evaluate(
    model,
    loader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str | Path,
):
    output_dir = Path(output_dir).expanduser().resolve()
    log_path = resolve_answers_log_path(output_dir, args.answers_log_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics, pred_records = evaluate_spatial_model(
        model,
        loader,
        device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        desc="eval",
        answers_log_path=log_path,
        validate_batch=_validate_generated_batch,
    )
    dump_json(all_metrics, output_dir / "metrics.json")
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(pred_records, f, ensure_ascii=False, indent=2)
    return all_metrics, pred_records


def run_evaluation(args: argparse.Namespace):
    args.test_json = str(ensure_foa_json(args.test_json, "--test-json"))
    main_device = torch.device(args.device)
    audio_encoder_device = torch.device(args.audio_encoder_device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    resolve_answers_log_path(output_dir, args.answers_log_name)
    output_dir = ensure_dir(output_dir)
    dump_json(vars(args), output_dir / "eval_args.json")

    dataset = build_dataset(args)
    loader = build_loader(dataset, args)
    model = build_model(args)
    load_required_checkpoint(model, args.checkpoint, args)
    model.to(main_device)
    model.backbone.audio_encoder.to(audio_encoder_device)
    model.eval()
    return evaluate(model, loader, args, main_device, output_dir)


def main(argv: Optional[Sequence[str]] = None):
    return run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
