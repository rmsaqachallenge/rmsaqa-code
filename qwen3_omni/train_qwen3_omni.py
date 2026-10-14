"""Phase 3 training entry-point helpers for Spatial Qwen3-Omni."""

from __future__ import annotations

import argparse
import copy
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from numbers import Real
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from tqdm.auto import tqdm

from spatial_af3.layered_dataset import SpatialQAPairDataset, qa_pair_collate_fn
from spatial_af3.layered_evaluation import evaluate_spatial_model, validate_generated_batch
from spatial_af3.metrics import STAGE2_CATEGORIES
from spatial_af3.models.spatial_qwen3_omni_generative import (
    SpatialQwen3OmniGenerativeModel,
)
from spatial_af3.utils import (
    cleanup_distributed,
    dump_json,
    ensure_dir,
    get_rank,
    is_main_process,
    seed_everything,
)


def ensure_foa_json(json_path: str | Path, arg_name: str) -> Path:
    del arg_name
    return Path(json_path).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure Spatial Qwen3-Omni generative training."
    )
    parser.add_argument("--qwen3-omni-root", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--val-feature-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-num-workers", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--val-max-records", type=int, default=None)
    parser.add_argument("--val-max-new-tokens", type=int, default=48)
    parser.add_argument("--val-num-beams", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--audio-token-stride", type=int, default=3)
    parser.add_argument("--llm-lora-r", type=int, default=12)
    parser.add_argument("--llm-lora-alpha", type=int, default=12)
    parser.add_argument("--llm-lora-dropout", type=float, default=0.05)
    parser.add_argument("--llm-lora-num-layers", type=int, default=48)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--extract-features-on-the-fly", action="store_true")
    parser.add_argument("--write-missing-features", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name, value in {
        "save-every-epoch": args.save_every_epoch,
        "extract-features-on-the-fly": args.extract_features_on_the_fly,
        "write-missing-features": args.write_missing_features,
    }.items():
        if type(value) is not bool:
            raise ValueError(f"--{name} must be a boolean.")

    integer_values = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "num-workers": args.num_workers,
        "val-batch-size": args.val_batch_size,
        "val-num-workers": args.val_num_workers,
        "max-records": args.max_records,
        "val-max-records": args.val_max_records,
        "val-max-new-tokens": args.val_max_new_tokens,
        "val-num-beams": args.val_num_beams,
        "grad-accum": args.grad_accum,
        "seed": args.seed,
        "audio-token-stride": args.audio_token_stride,
        "llm-lora-r": args.llm_lora_r,
        "llm-lora-alpha": args.llm_lora_alpha,
        "llm-lora-num-layers": args.llm_lora_num_layers,
        "task-id": args.task_id,
    }
    optional_integer_names = {"max-records", "val-max-records", "task-id"}
    for name, value in integer_values.items():
        if name in optional_integer_names and value is None:
            continue
        if type(value) is not int:
            raise ValueError(f"--{name} must be an integer.")

    for name, value in {
        "lr": args.lr,
        "weight-decay": args.weight_decay,
        "llm-lora-dropout": args.llm_lora_dropout,
    }.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"--{name} must be a finite real number.")

    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "val-batch-size": args.val_batch_size,
        "grad-accum": args.grad_accum,
        "val-max-new-tokens": args.val_max_new_tokens,
        "val-num-beams": args.val_num_beams,
        "llm-lora-r": args.llm_lora_r,
        "llm-lora-alpha": args.llm_lora_alpha,
        "llm-lora-num-layers": args.llm_lora_num_layers,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name} must be greater than zero.")

    for name, value in {
        "num-workers": args.num_workers,
        "val-num-workers": args.val_num_workers,
    }.items():
        if value < 0:
            raise ValueError(f"--{name} must be non-negative.")

    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and greater than zero.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative.")
    if not math.isfinite(args.llm_lora_dropout) or not (0 <= args.llm_lora_dropout < 1):
        raise ValueError("--llm-lora-dropout must be finite and in [0, 1).")
    if not 1 <= args.audio_token_stride <= 1500:
        raise ValueError("--audio-token-stride must be between 1 and 1500.")

    for name, value in {
        "max-records": args.max_records,
        "val-max-records": args.val_max_records,
    }.items():
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be greater than zero when provided.")

    if args.write_missing_features and not args.extract_features_on_the_fly:
        raise ValueError(
            "--write-missing-features requires --extract-features-on-the-fly."
        )


def _normalize_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    if not isinstance(args, argparse.Namespace):
        raise TypeError("run_training args must be an argparse.Namespace.")
    required_names = (
        "qwen3_omni_root",
        "feature_root",
        "val_json",
        "output_dir",
    )
    missing_required = [name for name in required_names if not hasattr(args, name)]
    if missing_required:
        raise ValueError(
            "run_training args are missing required fields: "
            + ", ".join(missing_required)
        )

    parser = build_parser()
    defaultable_names = (
        "train_json",
        "val_feature_root",
        "epochs",
        "batch_size",
        "num_workers",
        "val_batch_size",
        "val_num_workers",
        "max_records",
        "val_max_records",
        "val_max_new_tokens",
        "val_num_beams",
        "lr",
        "weight_decay",
        "grad_accum",
        "seed",
        "audio_token_stride",
        "llm_lora_r",
        "llm_lora_alpha",
        "llm_lora_dropout",
        "llm_lora_num_layers",
        "task_id",
        "init_checkpoint",
        "save_every_epoch",
        "extract_features_on_the_fly",
        "write_missing_features",
        "device",
    )
    for name in defaultable_names:
        if not hasattr(args, name):
            setattr(args, name, parser.get_default(name))
    if args.val_feature_root is None:
        args.val_feature_root = args.feature_root
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    if args.val_num_workers is None:
        args.val_num_workers = args.num_workers

    for name in (
        "save_every_epoch",
        "extract_features_on_the_fly",
        "write_missing_features",
    ):
        if type(getattr(args, name)) is not bool:
            raise ValueError(f"{name} must be a boolean.")

    for name in (
        "qwen3_omni_root",
        "train_json",
        "feature_root",
        "val_json",
        "val_feature_root",
        "output_dir",
        "init_checkpoint",
    ):
        value = getattr(args, name)
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"{name} must be a string or os.PathLike path.")
        normalized_path = os.fspath(value)
        if type(normalized_path) is not str:
            raise TypeError(f"{name} must resolve to a string path.")
        setattr(args, name, normalized_path)
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.val_feature_root is None:
        args.val_feature_root = args.feature_root
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    if args.val_num_workers is None:
        args.val_num_workers = args.num_workers
    validate_args(args)
    return args


def resolve_device(args: argparse.Namespace) -> tuple[torch.device, Optional[int]]:
    try:
        requested_device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        raw_device = str(args.device)
        if not raw_device.startswith("cuda:"):
            raise
        raw_index = raw_device.partition(":")[2]
        try:
            invalid_index = int(raw_index)
        except ValueError:
            raise ValueError(f"Invalid CUDA device index: {raw_index!r}.") from error
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.") from error
        device_count = int(torch.cuda.device_count())
        raise ValueError(
            f"CUDA device index {invalid_index} is outside [0, {device_count})."
        ) from error
    if requested_device.type == "cpu":
        return torch.device("cpu"), None
    if requested_device.type != "cuda":
        return requested_device, None
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    device_count = int(torch.cuda.device_count())
    if requested_device.index is not None:
        local_rank = requested_device.index
        if local_rank < 0 or local_rank >= device_count:
            raise ValueError(
                f"CUDA device index {local_rank} is outside [0, {device_count})."
            )
    else:
        environment_local_rank = os.environ.get("LOCAL_RANK")
        try:
            launcher_rank = int(environment_local_rank)
        except (TypeError, ValueError):
            launcher_rank = None
        if launcher_rank is not None and 0 <= launcher_rank < device_count:
            local_rank = launcher_rank
        else:
            local_rank = int(torch.cuda.current_device())
    return torch.device("cuda", local_rank), local_rank


def init_training_distributed(
    device: torch.device,
    local_rank: Optional[int],
) -> bool:
    if dist.is_initialized():
        return True
    launcher_variables = (
        os.environ.get("RANK"),
        os.environ.get("WORLD_SIZE"),
        os.environ.get("LOCAL_RANK"),
        os.environ.get("MASTER_ADDR"),
        os.environ.get("MASTER_PORT"),
    )
    if any(value is None for value in launcher_variables):
        return False
    if device.type == "cuda":
        if local_rank is None:
            raise ValueError("local_rank is required for CUDA distributed training.")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True


def synchronize_main_process_error(
    error: Optional[BaseException],
    *,
    is_distributed: bool,
    device: torch.device,
    main_process: bool,
    stage: str,
) -> None:
    if not is_distributed:
        if error is not None:
            raise RuntimeError(f"{stage} failed.") from error
        return

    status = torch.tensor(
        0 if main_process and error is not None else 1,
        dtype=torch.int32,
        device=device,
    )
    dist.broadcast(status, src=0)
    if int(status.item()) != 0:
        return
    if error is not None:
        raise error
    raise RuntimeError(f"Rank 0 failed during {stage}.")


def synchronize_rank_success(local_success: bool, device: torch.device) -> bool:
    success = torch.tensor(int(local_success), dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(success, op=dist.ReduceOp.MIN)
    return bool(success.item())


def raise_on_rank_failure(
    local_error: Optional[BaseException],
    *,
    device: torch.device,
    stage: str,
) -> None:
    if synchronize_rank_success(local_error is None, device):
        return
    if local_error is not None:
        if dist.is_available() and dist.is_initialized():
            failure_location = f" on rank {dist.get_rank()}"
        else:
            failure_location = ""
        raise RuntimeError(f"{stage} failed{failure_location}.") from local_error
    raise RuntimeError(f"{stage} failed on another rank.")


class ForwardTrainAdapter(nn.Module):
    def __init__(self, training_model: nn.Module) -> None:
        super().__init__()
        self.training_model = training_model

    def forward(self, *args, **kwargs):
        return self.training_model.forward_train(*args, **kwargs)


def call_forward_train(model: nn.Module, batch, device: torch.device):
    return model(batch, device=device, enable_layer2=True)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: Sequence[nn.Parameter],
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
    show_progress: bool,
) -> tuple[dict[str, float], int]:
    def raise_synchronized(local_error: Optional[BaseException], stage: str) -> None:
        if synchronize_rank_success(local_error is None, device):
            return
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"{stage} failed on another rank.")

    setup_error: Optional[BaseException] = None
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        parameters = list(trainable_parameters)
        if not torch.is_grad_enabled():
            raise RuntimeError("train_one_epoch requires a gradient-enabled context.")
        if not parameters:
            raise ValueError("Training requires non-empty trainable parameters.")
        seen_parameter_ids: set[int] = set()
        for index, parameter in enumerate(parameters):
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(
                    f"trainable_parameters[{index}] must be an nn.Parameter."
                )
            if id(parameter) in seen_parameter_ids:
                raise ValueError(
                    f"trainable_parameters contains duplicate parameter at index {index}."
                )
            if not parameter.requires_grad:
                raise ValueError(
                    f"trainable_parameters[{index}] must have requires_grad=True."
                )
            seen_parameter_ids.add(id(parameter))
        expected_batches = len(loader)
        if type(expected_batches) is not int or expected_batches <= 0:
            raise ValueError(
                "Training loader is empty or does not report a positive batch count."
            )
        if type(grad_accum) is not int or grad_accum <= 0:
            raise ValueError("grad_accum must be a positive integer.")
    except Exception as error:
        setup_error = error
        parameters = []
        expected_batches = 0
    raise_synchronized(setup_error, "training setup")

    if dist.is_available() and dist.is_initialized():
        gathered_batch_counts: list[object] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_batch_counts, expected_batches)
        if any(count != expected_batches for count in gathered_batch_counts):
            raise RuntimeError(
                "Training loader batch counts differ across ranks: "
                f"{gathered_batch_counts}."
            )

    metric_sums: dict[str, float] = {}
    expected_metric_names: Optional[tuple[str, ...]] = None
    batch_count = 0
    window_batches = 0
    optimizer_steps = 0
    iterator_error: Optional[BaseException] = None
    try:
        iterable = (
            tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}")
            if show_progress
            else loader
        )
        iterator = iter(iterable)
    except Exception as error:
        iterator_error = error
        iterator = iter(())
    raise_synchronized(iterator_error, "training progress/iterator setup")

    def finish_optimizer_step() -> None:
        nonlocal optimizer_steps, window_batches
        gradient_target_error: Optional[BaseException] = None
        if not any(parameter.grad is not None for parameter in parameters):
            gradient_target_error = ValueError(
                "Training loss produced no gradients for trainable parameters."
            )
        raise_synchronized(gradient_target_error, "training target gradient validation")
        correction_error: Optional[BaseException] = None
        try:
            if window_batches < grad_accum:
                correction = grad_accum / window_batches
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
        except Exception as error:
            correction_error = error
        raise_synchronized(correction_error, "training tail gradient correction")
        clip_error: Optional[BaseException] = None
        try:
            torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        except Exception as error:
            clip_error = error
        raise_synchronized(clip_error, "training gradient validation")
        try:
            optimizer.step()
        except Exception as error:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            raise RuntimeError(
                f"training optimizer step failed fatally: {error}"
            ) from error
        optimizer_steps += 1
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception as error:
            raise RuntimeError(
                f"training post-step gradient cleanup failed fatally: {error}"
            ) from error
        window_batches = 0

    for batch_index in range(expected_batches):
        fetch_error: Optional[BaseException] = None
        try:
            batch = next(iterator)
        except StopIteration:
            fetch_error = RuntimeError(
                "Training loader batch count mismatch: "
                f"reported {expected_batches} but produced {batch_index}."
            )
            batch = None
        except Exception as error:
            fetch_error = error
            batch = None
        raise_synchronized(fetch_error, "training batch fetch")

        context_error: Optional[BaseException] = None
        sync_context = nullcontext()
        context_entered = False
        try:
            should_defer_sync = (
                window_batches + 1 < grad_accum and batch_index + 1 < expected_batches
            )
            no_sync_factory = (
                getattr(model, "no_sync", None) if should_defer_sync else None
            )
            use_no_sync = callable(no_sync_factory)
            sync_context = no_sync_factory() if use_no_sync else nullcontext()
            sync_context.__enter__()
            context_entered = True
        except Exception as error:
            context_error = error
        global_context_valid = synchronize_rank_success(context_error is None, device)
        if not global_context_valid:
            cleanup_error: Optional[BaseException] = None
            if context_entered:
                try:
                    sync_context.__exit__(None, None, None)
                except Exception as error:
                    cleanup_error = error
                context_entered = False
            if cleanup_error is not None:
                raise RuntimeError(
                    "training no_sync setup failed and context cleanup failed "
                    f"fatally: {cleanup_error}"
                ) from (context_error or cleanup_error)
            if context_error is not None:
                raise context_error
            raise RuntimeError("training no_sync context setup failed on another rank.")

        validation_error: Optional[BaseException] = None
        loss: Optional[torch.Tensor] = None
        validated_metrics: dict[str, float] = {}
        try:
            output = call_forward_train(model, batch, device)
            if not isinstance(output, Mapping):
                raise TypeError("Training output must be a mapping.")
            if "loss" not in output:
                raise ValueError("Training output field 'loss' is required.")
            loss_value = output["loss"]
            if not isinstance(loss_value, torch.Tensor):
                raise TypeError("Training output field 'loss' must be a scalar tensor.")
            if loss_value.ndim != 0:
                raise ValueError(
                    "Training output field 'loss' must be a scalar tensor."
                )
            if torch.is_complex(loss_value):
                raise TypeError(
                    "Training output field 'loss' must be real, not complex."
                )
            if not loss_value.is_floating_point():
                raise TypeError(
                    "Training output field 'loss' must have floating dtype."
                )
            if torch.is_grad_enabled() and not loss_value.requires_grad:
                raise ValueError(
                    "Training output field 'loss' must require a gradient."
                )

            if any(not isinstance(name, str) for name in output):
                raise TypeError("Training output metric keys must be strings.")
            metric_names = tuple(sorted(output))
            if (
                expected_metric_names is not None
                and metric_names != expected_metric_names
            ):
                raise ValueError(
                    "Training output metric keys changed between batches: "
                    f"expected {expected_metric_names}, got {metric_names}."
                )
            tensor_metric_names: list[str] = []
            tensor_metric_values: list[torch.Tensor] = []
            for name in metric_names:
                value = output[name]
                if isinstance(value, torch.Tensor):
                    if value.ndim != 0:
                        raise ValueError(
                            f"Training output field '{name}' must be scalar."
                        )
                    if torch.is_complex(value):
                        raise TypeError(
                            f"Training output field '{name}' must be real, not complex."
                        )
                    if value.dtype == torch.bool:
                        raise TypeError(
                            f"Training output field '{name}' must be a real number."
                        )
                    tensor_metric_names.append(name)
                    tensor_metric_values.append(value)
                    continue
                else:
                    scalar = value
                if isinstance(scalar, bool) or not isinstance(scalar, Real):
                    raise TypeError(
                        f"Training output field '{name}' must be a real scalar."
                    )
                scalar_float = float(scalar)
                if not math.isfinite(scalar_float):
                    raise ValueError(f"Training output field '{name}' must be finite.")
                validated_metrics[name] = scalar_float
            try:
                tensor_metric_scalars = (
                    torch.stack(
                        [
                            value.detach().to(
                                device=loss_value.device, dtype=torch.float64
                            )
                            for value in tensor_metric_values
                        ]
                    )
                    .cpu()
                    .tolist()
                )
            except Exception as error:
                raise ValueError(
                    "Training output tensor metrics could not be converted to scalars."
                ) from error
            for name, scalar in zip(tensor_metric_names, tensor_metric_scalars):
                scalar_float = float(scalar)
                if not math.isfinite(scalar_float):
                    raise ValueError(f"Training output field '{name}' must be finite.")
                validated_metrics[name] = scalar_float
            loss = loss_value
        except Exception as error:
            validation_error = error
        global_output_valid = synchronize_rank_success(validation_error is None, device)
        if not global_output_valid:
            cleanup_error = None
            try:
                if context_entered:
                    sync_context.__exit__(
                        (
                            type(validation_error)
                            if validation_error is not None
                            else None
                        ),
                        validation_error,
                        (
                            validation_error.__traceback__
                            if validation_error is not None
                            else None
                        ),
                    )
            except Exception as error:
                cleanup_error = error
            context_entered = False
            if cleanup_error is not None:
                raise RuntimeError(
                    "training output validation failed and no_sync context cleanup "
                    f"failed fatally: {cleanup_error}"
                ) from (validation_error or cleanup_error)
            if validation_error is not None:
                raise validation_error
            raise RuntimeError("training output validation failed on another rank.")

        if expected_metric_names is None:
            expected_metric_names = tuple(sorted(validated_metrics))
        for name, scalar in validated_metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + scalar
        assert loss is not None
        try:
            (loss / grad_accum).backward()
        except Exception as error:
            try:
                if context_entered:
                    sync_context.__exit__(type(error), error, error.__traceback__)
            except Exception:
                pass
            context_entered = False
            raise RuntimeError(f"training backward failed fatally: {error}") from error
        try:
            if context_entered:
                sync_context.__exit__(None, None, None)
        except Exception as error:
            raise RuntimeError(
                f"training no_sync context exit failed fatally: {error}"
            ) from error
        context_entered = False
        batch_count += 1
        window_batches += 1
        if window_batches == grad_accum:
            finish_optimizer_step()

    extra_error: Optional[BaseException] = None
    try:
        next(iterator)
    except StopIteration:
        pass
    except Exception as error:
        extra_error = error
    else:
        extra_error = RuntimeError(
            "Training loader batch count mismatch: "
            f"reported {expected_batches} but produced more batches."
        )
    raise_synchronized(extra_error, "training loader batch count validation")

    if window_batches:
        finish_optimizer_step()

    metric_names = tuple(sorted(metric_sums))
    if dist.is_available() and dist.is_initialized():
        gathered_metadata: list[object] = [None] * dist.get_world_size()
        local_metadata = (metric_names, optimizer_steps)
        dist.all_gather_object(gathered_metadata, local_metadata)
        if any(metadata != local_metadata for metadata in gathered_metadata):
            raise RuntimeError(
                "Training metric keys or optimizer step counts differ across ranks."
            )
        try:
            totals = torch.tensor(
                [*(metric_sums[name] for name in metric_names), float(batch_count)],
                dtype=torch.float64,
                device=device,
            )
        except Exception as error:
            raise RuntimeError(
                "training metric totals construction failed fatally: " f"{error}"
            ) from error
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        global_batch_count = float(totals[-1].item())
        metric_sums = {
            name: float(totals[index].item()) for index, name in enumerate(metric_names)
        }
    else:
        global_batch_count = float(batch_count)
    return (
        {name: metric_sums[name] / global_batch_count for name in metric_names},
        optimizer_steps,
    )


def wrap_ddp(
    model: nn.Module,
    is_distributed: bool,
    device: torch.device,
    local_rank: Optional[int],
) -> nn.Module:
    if not is_distributed:
        return model
    if device.type == "cuda":
        if local_rank is None:
            raise ValueError("local_rank is required for CUDA DDP.")
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    return DDP(model, find_unused_parameters=False)


def unwrap_model(model: nn.Module) -> nn.Module:
    visited_ids: set[int] = set()
    while True:
        if id(model) in visited_ids:
            raise ValueError("Model wrapper cycle detected while unwrapping.")
        visited_ids.add(id(model))
        if isinstance(model, DDP):
            model = model.module
        elif isinstance(model, ForwardTrainAdapter):
            model = model.training_model
        else:
            return model


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Model has no trainable parameters.")
    return parameters


def _trainable_state_names(
    model: nn.Module,
) -> tuple[nn.Module, tuple[str, ...], tuple[str, ...]]:
    core = unwrap_model(model)
    trainable_names = {
        name for name, parameter in core.named_parameters() if parameter.requires_grad
    }
    buffer_names: set[str] = set()
    for module_name, module in core.named_modules():
        has_direct_trainable = any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        )
        if not has_direct_trainable:
            continue
        for buffer_name, _buffer in module.named_buffers(recurse=False):
            if buffer_name in module._non_persistent_buffers_set:
                continue
            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            buffer_names.add(full_name)
    return core, tuple(sorted(trainable_names)), tuple(sorted(buffer_names))


def synchronize_trainable_buffers_for_validation(
    model: nn.Module,
    *,
    device: torch.device,
) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return

    core: Optional[nn.Module] = None
    buffer_items: list[tuple[str, torch.Tensor]] = []
    validation_error: Optional[BaseException] = None
    try:
        core, _trainable_names, buffer_names = _trainable_state_names(model)
        complete_state = core.state_dict(keep_vars=True)
        if not isinstance(complete_state, Mapping):
            raise TypeError("Model state_dict must be a mapping.")
        for name in buffer_names:
            if name not in complete_state:
                raise ValueError(
                    "Trainable buffer state_dict is missing required name: " + name
                )
            buffer = complete_state[name]
            if type(buffer) is not torch.Tensor:
                raise TypeError(
                    f"Trainable buffer '{name}' must be an exact torch.Tensor."
                )
            if buffer.device.type == "meta":
                raise ValueError(
                    f"Trainable buffer '{name}' must contain materialized values."
                )
            if buffer.layout != torch.strided:
                raise ValueError(f"Trainable buffer '{name}' must use strided layout.")
            buffer_items.append((name, buffer))
    except BaseException as error:
        validation_error = error
    raise_on_rank_failure(
        validation_error,
        device=device,
        stage="trainable buffer validation",
    )

    metadata = tuple(
        (name, tuple(buffer.shape), str(buffer.dtype)) for name, buffer in buffer_items
    )
    gathered_metadata: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_metadata, metadata)
    if any(
        rank_metadata != gathered_metadata[0] for rank_metadata in gathered_metadata
    ):
        raise RuntimeError(
            "Trainable buffer names, shapes, or dtypes differ across ranks."
        )
    for _name, buffer in buffer_items:
        dist.broadcast(buffer, src=0)


def load_initial_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    if isinstance(checkpoint_path, str) and not checkpoint_path.strip():
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Initial checkpoint payload must be a mapping.")
    if "state_dict" in payload:
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError("Initial checkpoint 'state_dict' must be a mapping.")
    else:
        state_dict = payload
    if not state_dict:
        raise ValueError("Initial checkpoint state_dict must not be empty.")
    core, trainable_names, buffer_names = _trainable_state_names(model)
    allowed_names = set(trainable_names)
    allowed_names.update(buffer_names)
    model_state = core.state_dict()
    matching_names = allowed_names.intersection(model_state).intersection(state_dict)
    if not matching_names:
        raise ValueError(
            "Initial checkpoint state_dict has no allowed trainable state keys "
            "matching the model."
        )
    filtered_state_dict = {name: state_dict[name] for name in sorted(matching_names)}
    core.load_state_dict(filtered_state_dict, strict=False)


def save_trainable_checkpoint(
    model: nn.Module,
    out_path: Path,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    extra_state: Optional[dict[str, object]] = None,
) -> None:
    def validate_weights_only_value(value: object, field_path: str) -> None:
        value_type = type(value)
        if value_type is torch.Tensor or value is None:
            return
        if value_type in (bool, int, float, str):
            return
        if value_type in (list, tuple):
            for index, item in enumerate(value):
                validate_weights_only_value(item, f"{field_path}[{index}]")
            return
        if value_type is dict:
            for key, item in value.items():
                if key is not None and type(key) not in (bool, int, float, str):
                    raise TypeError(
                        f"Checkpoint field '{field_path}[<key>]' is not "
                        "weights_only-safe."
                    )
                if type(key) is str:
                    item_path = f"{field_path}.{key}"
                else:
                    item_path = f"{field_path}[{key}]"
                validate_weights_only_value(item, item_path)
            return
        raise TypeError(f"Checkpoint field '{field_path}' is not weights_only-safe.")

    core, trainable_names, buffer_names = _trainable_state_names(model)
    if not trainable_names:
        raise ValueError("Model has no trainable parameters to save.")

    selected_names = set(trainable_names)
    selected_names.update(buffer_names)

    complete_state = core.state_dict()
    if not isinstance(complete_state, Mapping):
        raise TypeError("Model state_dict must be a mapping.")
    missing = sorted(selected_names - set(complete_state))
    if missing:
        raise ValueError(
            "Trainable checkpoint state_dict is missing required names: "
            + ", ".join(missing)
        )

    state_dict: dict[str, torch.Tensor] = {}
    for name in sorted(selected_names):
        value = complete_state[name]
        if type(value) is not torch.Tensor:
            raise TypeError(
                f"Checkpoint state_dict field '{name}' must be an exact torch.Tensor."
            )
        state_dict[name] = value.detach().cpu().clone()

    core_keys = {"state_dict", "args", "epoch", "step"}
    if extra_state is not None:
        if not isinstance(extra_state, dict):
            raise TypeError("extra_state must be a dict when provided.")
        conflicts = sorted(core_keys.intersection(extra_state))
        if conflicts:
            raise ValueError(
                "extra_state cannot override core checkpoint key(s): "
                + ", ".join(conflicts)
            )

    args_state = vars(args).copy()
    validate_weights_only_value(args_state, "args")
    validate_weights_only_value(epoch, "epoch")
    validate_weights_only_value(step, "step")
    if extra_state is not None:
        validate_weights_only_value(extra_state, "extra_state")

    payload: dict[str, object] = {
        "state_dict": state_dict,
        "args": args_state,
        "epoch": epoch,
        "step": step,
    }
    if extra_state is not None:
        payload.update(extra_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, out_path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _raise_sample_error(split: str, field: str, detail: str) -> None:
    raise ValueError(f"{split} sample field '{field}' {detail}")


def _validate_feature_tensor(
    sample: Mapping[str, object],
    *,
    field: str,
    split: str,
) -> torch.Tensor:
    value = sample.get(field)
    if not isinstance(value, torch.Tensor):
        _raise_sample_error(split, field, "must be a torch.Tensor.")
    if value.layout != torch.strided:
        _raise_sample_error(split, field, "must use strided layout.")
    if value.device.type == "meta":
        _raise_sample_error(split, field, "must contain materialized values.")
    if value.is_quantized:
        _raise_sample_error(split, field, "must not be quantized.")
    supported_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if value.dtype not in supported_dtypes:
        _raise_sample_error(
            split,
            field,
            "must use float16, bfloat16, float32, or float64 dtype.",
        )
    if not bool(torch.isfinite(value).all()):
        _raise_sample_error(split, field, "must contain only finite values.")
    return value


def validate_qwen3_dataset_sample(sample: Mapping[str, object], split: str) -> None:
    if not isinstance(sample, Mapping):
        _raise_sample_error(split, "sample", "must be a mapping.")

    whisper = _validate_feature_tensor(
        sample,
        field="whisper_input_features",
        split=split,
    )
    if tuple(whisper.shape) != (128, 1000):
        _raise_sample_error(
            split,
            "whisper_input_features",
            "must have shape [128, 1000].",
        )

    spatial = _validate_feature_tensor(
        sample,
        field="spatial_features",
        split=split,
    )
    if spatial.ndim != 2 or spatial.shape[0] <= 0 or spatial.shape[1] != 448:
        _raise_sample_error(
            split,
            "spatial_features",
            "must have shape [T, 448] with T greater than zero.",
        )

    metadata_fields = (
        "segment_id",
        "audio_path",
        "layer2_category_id",
        "layer2_category_name",
        "task_id",
    )
    for field in metadata_fields:
        if field not in sample:
            _raise_sample_error(split, field, "is required.")

    for field in ("segment_id", "audio_path"):
        value = sample[field]
        if not isinstance(value, str) or not value.strip():
            _raise_sample_error(split, field, "must be a non-empty string.")

    category_id = sample["layer2_category_id"]
    category_name = sample["layer2_category_name"]
    if split in ("validation", "val"):
        if type(category_id) is not int or category_id not in STAGE2_CATEGORIES:
            _raise_sample_error(
                split,
                "layer2_category_id",
                "must be an exact supported Layer 2 category integer ID.",
            )
        if type(category_name) is not str or not category_name:
            _raise_sample_error(
                split,
                "layer2_category_name",
                "must be the exact non-empty canonical Layer 2 category name.",
            )
        if category_name != STAGE2_CATEGORIES[category_id]:
            _raise_sample_error(
                split,
                "layer2_category_name",
                "must exactly match the canonical name for layer2_category_id.",
            )
    else:
        if category_id is not None and type(category_id) is not int:
            _raise_sample_error(
                split,
                "layer2_category_id",
                "must be an integer or None.",
            )
        if category_name is not None and (
            not isinstance(category_name, str) or not category_name.strip()
        ):
            _raise_sample_error(
                split,
                "layer2_category_name",
                "must be a non-empty string or None.",
            )

    task_id = sample["task_id"]
    if task_id is not None and type(task_id) is not int:
        _raise_sample_error(split, "task_id", "must be an integer or None.")

    for layer in (1, 2):
        for suffix in ("question", "answer"):
            field = f"layer{layer}_{suffix}"
            value = sample.get(field)
            if not isinstance(value, str) or not value.strip():
                _raise_sample_error(split, field, "must be a non-empty string.")

        options_field = f"layer{layer}_options"
        options = sample.get(options_field)
        if not isinstance(options, Mapping) or not options:
            _raise_sample_error(
                split,
                options_field,
                "must be a non-empty Mapping[str, str].",
            )
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in options.items()
        ):
            _raise_sample_error(
                split,
                options_field,
                "must be a non-empty Mapping[str, str].",
            )

        correct_field = f"layer{layer}_correct_option"
        correct = sample.get(correct_field)
        if not isinstance(correct, str) or not correct.strip():
            plural_field = f"layer{layer}_correct_options"
            plural_note = (
                f"; plural alias '{plural_field}' is not supported"
                if plural_field in sample
                else ""
            )
            _raise_sample_error(
                split,
                correct_field,
                f"must be a non-empty string{plural_note}.",
            )
        if correct not in options:
            _raise_sample_error(
                split,
                correct_field,
                f"must name a key present in '{options_field}'.",
            )

    try:
        batch = qa_pair_collate_fn([sample])
    except Exception as error:
        field = (
            str(error.args[0])
            if isinstance(error, KeyError) and error.args
            else "collate"
        )
        raise ValueError(
            f"{split} sample field '{field}' failed during collation."
        ) from error
    for field, tensor in (
        ("whisper_input_features", whisper),
        ("spatial_features", spatial),
    ):
        batched = batch.get(field) if isinstance(batch, Mapping) else None
        if not isinstance(batched, torch.Tensor):
            _raise_sample_error(
                split,
                field,
                "must be a torch.Tensor after collation.",
            )
        if batched.layout != torch.strided:
            _raise_sample_error(
                split,
                field,
                "must use strided layout after collation.",
            )
        if batched.device.type == "meta":
            _raise_sample_error(
                split,
                field,
                "must not use the meta device after collation.",
            )
        if batched.is_quantized:
            _raise_sample_error(
                split,
                field,
                "must not be quantized after collation.",
            )
        if batched.device != tensor.device:
            _raise_sample_error(
                split,
                field,
                "must preserve device after collation.",
            )
        if batched.dtype != tensor.dtype:
            _raise_sample_error(
                split,
                field,
                "must preserve dtype after collation.",
            )
        expected_shape = (1, *tensor.shape)
        if tuple(batched.shape) != tuple(expected_shape):
            _raise_sample_error(
                split,
                field,
                f"must gain only a leading batch axis after collation; expected "
                f"{tuple(expected_shape)}.",
            )
        try:
            values_match = torch.equal(batched[0], tensor)
        except Exception as error:
            raise ValueError(
                f"{split} sample field '{field}' comparison failed after " "collation."
            ) from error
        if not values_match:
            _raise_sample_error(
                split,
                field,
                "must preserve values after adding the leading batch axis.",
            )


def build_dataset(args: argparse.Namespace) -> SpatialQAPairDataset:
    dataset = SpatialQAPairDataset(
        json_path=args.train_json,
        feature_root=args.feature_root,
        extract_features_on_the_fly=args.extract_features_on_the_fly,
        write_missing_features=args.write_missing_features,
        audio_token_stride=args.audio_token_stride,
        max_records=args.max_records,
        task_id=args.task_id,
    )
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty.")
    validate_qwen3_dataset_sample(dataset[0], "training")
    return dataset


def build_validation_dataset(args: argparse.Namespace) -> SpatialQAPairDataset:
    dataset = SpatialQAPairDataset(
        json_path=args.val_json,
        feature_root=args.val_feature_root,
        extract_features_on_the_fly=args.extract_features_on_the_fly,
        write_missing_features=args.write_missing_features,
        audio_token_stride=args.audio_token_stride,
        max_records=args.val_max_records,
        task_id=args.task_id,
    )
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty.")
    validate_qwen3_dataset_sample(dataset[0], "validation")
    return dataset


def build_loader(
    dataset: SpatialQAPairDataset,
    args: argparse.Namespace,
    is_distributed: bool,
) -> tuple[DataLoader, object]:
    sampler = (
        DistributedSampler(dataset, shuffle=True)
        if is_distributed
        else RandomSampler(dataset)
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=qa_pair_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return loader, sampler


class DistributedEvalSampler(Sampler[int]):
    """Shard validation indices across ranks without padding duplicates."""

    def __init__(
        self,
        dataset,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError(
                    "num_replicas is required when distributed is not initialized."
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError(
                    "rank is required when distributed is not initialized."
                )
            rank = dist.get_rank()
        if type(num_replicas) is not int or num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer.")
        if type(rank) is not int or rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be an integer in [0, num_replicas).")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        dataset_len = len(self.dataset)
        if dataset_len <= self.rank:
            return 0
        return ((dataset_len - 1 - self.rank) // self.num_replicas) + 1


def build_validation_loader(
    dataset: SpatialQAPairDataset,
    args: argparse.Namespace,
    is_distributed: bool,
) -> DataLoader:
    sampler = (
        DistributedEvalSampler(dataset)
        if is_distributed
        else SequentialSampler(dataset)
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.val_batch_size,
        num_workers=args.val_num_workers,
        collate_fn=qa_pair_collate_fn,
        pin_memory=True,
        drop_last=False,
    )


def validation_metrics_to_count_tensor(metrics, *, device):
    if not isinstance(metrics, Mapping):
        raise TypeError("Validation metrics must be a mapping.")
    if set(metrics) != {"overall", "categories"}:
        raise ValueError("Validation metrics must contain exactly 'overall' and 'categories'.")
    categories = metrics["categories"]
    if not isinstance(categories, Mapping):
        raise TypeError("Validation metrics 'categories' entry must be a mapping.")
    expected_keys = {str(category_id) for category_id in STAGE2_CATEGORIES}
    if set(categories) != expected_keys:
        raise ValueError("Validation metric category keys do not match AF3 categories.")

    fields = ("stage1_acc", "stage2_acc_raw", "grounded_stage2_acc")
    buckets = [("overall", metrics["overall"])]
    if not isinstance(metrics["overall"], Mapping):
        raise TypeError("Validation metrics 'overall' entry must be a mapping.")
    for category_id, category_name in STAGE2_CATEGORIES.items():
        key = str(category_id)
        bucket = categories[key]
        expected = set(fields) | {"num_records", "category_id", "category_name"}
        if not isinstance(bucket, Mapping) or set(bucket) != expected:
            raise ValueError(f"Validation metric category {key} fields do not match AF3 schema.")
        if bucket["category_id"] != category_id or bucket["category_name"] != category_name:
            raise ValueError(f"Validation metric category {key} metadata is invalid.")
        buckets.append((f"category {key}", bucket))
    overall_expected = set(fields) | {"num_records"}
    if set(metrics["overall"]) != overall_expected:
        raise ValueError("Validation metric overall fields do not match AF3 schema.")

    counts = []
    for name, bucket in buckets:
        n = bucket["num_records"]
        if type(n) is not int or n < 0 or n > 2**53:
            raise ValueError(f"Validation metric {name} has invalid num_records.")
        recovered = []
        for field in fields:
            value = bucket[field]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Validation metric {name}.{field} must be a finite accuracy in [0, 1].")
            raw = float(value) * n
            rounded = round(raw)
            if not math.isclose(raw, rounded, rel_tol=0.0, abs_tol=max(1e-6, 4 * math.ulp(raw))):
                raise ValueError(f"Validation metric {name}.{field} does not encode an integer count.")
            recovered.append(int(rounded))
        stage1, stage2_raw, grounded = recovered
        if grounded > min(stage1, stage2_raw):
            raise ValueError(f"Validation metric {name} grounded count exceeds a component count.")
        counts.extend((n, *recovered))
    for index in range(4):
        if counts[index] != sum(counts[bucket * 4 + index] for bucket in range(1, 1 + len(STAGE2_CATEGORIES))):
            raise ValueError("Validation metric overall counts must equal the sum of category counts.")
    return torch.tensor(counts, dtype=torch.int64, device=device)



def validation_count_tensor_to_metrics(count_tensor):
    if not isinstance(count_tensor, torch.Tensor) or count_tensor.dtype != torch.int64:
        raise TypeError("Validation counts must be an int64 torch.Tensor.")
    expected_shape = (4 * (1 + len(STAGE2_CATEGORIES)),)
    if tuple(count_tensor.shape) != expected_shape:
        raise ValueError(f"Validation count tensor shape must be {expected_shape}.")
    values = [int(v) for v in count_tensor.detach().cpu().tolist()]
    if any(v < 0 for v in values):
        raise ValueError("Validation count tensor values must be non-negative.")
    buckets = []
    for index in range(1 + len(STAGE2_CATEGORIES)):
        offset = 4 * index
        n, stage1, stage2_raw, grounded = values[offset:offset+4]
        if n > 2**53 or any(v > n for v in (stage1, stage2_raw, grounded)):
            raise ValueError(f"Validation count bucket {index} has invalid counts.")
        if grounded > min(stage1, stage2_raw):
            raise ValueError(f"Validation count bucket {index} grounded count is invalid.")
        denom = max(n, 1)
        buckets.append({
            "num_records": n,
            "stage1_acc": stage1 / denom,
            "stage2_acc_raw": stage2_raw / denom,
            "grounded_stage2_acc": grounded / denom,
        })
    for field_index in range(4):
        if values[field_index] != sum(values[bucket * 4 + field_index] for bucket in range(1, 1 + len(STAGE2_CATEGORIES))):
            raise ValueError("Validation metric overall counts must equal the sum of category counts.")
    categories = {
        str(category_id): {"category_id": category_id, "category_name": name, **buckets[index]}
        for index, (category_id, name) in enumerate(STAGE2_CATEGORIES.items(), start=1)
    }
    return {"overall": buckets[0], "categories": categories}



def reduce_validation_metrics(
    metrics: Mapping[str, object],
    *,
    device: torch.device,
) -> dict[str, object]:
    local_error: Optional[BaseException] = None
    counts: Optional[torch.Tensor] = None
    try:
        counts = validation_metrics_to_count_tensor(metrics, device=device)
    except BaseException as error:
        local_error = error
    raise_on_rank_failure(
        local_error,
        device=device,
        stage="validation metric conversion",
    )
    assert counts is not None
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return validation_count_tensor_to_metrics(counts)


def validation_selection_score(metrics: Mapping[str, object]) -> float:
    if not isinstance(metrics, Mapping):
        raise TypeError("Validation metrics must be a mapping.")
    overall = metrics.get("overall")
    if not isinstance(overall, Mapping):
        raise TypeError("Validation metrics 'overall' entry must be a mapping.")
    score = overall.get("grounded_stage2_acc")
    if (
        isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(score)
    ):
        raise ValueError("Validation selection score must be a finite real number.")
    score_value = float(score)
    if not 0.0 <= score_value <= 1.0:
        raise ValueError("Validation selection score must be in [0, 1].")
    return score_value


def should_save_best(
    metrics: Mapping[str, object],
    best_score: Optional[float],
) -> bool:
    score = validation_selection_score(metrics)
    if best_score is None:
        return True
    if isinstance(best_score, bool) or not isinstance(best_score, Real):
        raise TypeError("best_score must be a real number or None.")
    best_score_value = float(best_score)
    if not math.isfinite(best_score_value):
        raise ValueError("best_score must be finite.")
    if not 0.0 <= best_score_value <= 1.0:
        raise ValueError("best_score must be in [0, 1].")
    return score > best_score_value


def evaluate_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    args: argparse.Namespace,
    main_process: bool,
) -> Mapping[str, object]:
    core = unwrap_model(model)
    metrics, _ = evaluate_spatial_model(
        core,
        loader,
        device,
        max_new_tokens=args.val_max_new_tokens,
        num_beams=args.val_num_beams,
        desc="Validation",
        collect_predictions=False,
        show_progress=main_process,
        validate_batch=validate_generated_batch,
    )
    if not isinstance(metrics, Mapping):
        raise TypeError("Validation evaluator metrics must be a mapping.")
    return metrics


def save_validation_artifacts(
    *,
    model: nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    metrics: Mapping[str, object],
    history: list[dict[str, object]],
    best_score: Optional[float],
) -> tuple[float, bool]:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        raise RuntimeError("Validation artifacts may only be saved on rank 0.")
    if type(history) is not list:
        raise TypeError("Validation history must be a list.")
    if type(epoch) is not int:
        raise TypeError("Validation artifact epoch must be an integer.")
    if epoch <= 0:
        raise ValueError("Validation artifact epoch must be positive.")
    if type(step) is not int:
        raise TypeError("Validation artifact step must be an integer.")
    if step < 0:
        raise ValueError("Validation artifact step must be non-negative.")
    metric_counts = validation_metrics_to_count_tensor(
        metrics,
        device=torch.device("cpu"),
    )
    metrics_snapshot = validation_count_tensor_to_metrics(metric_counts)
    canonical_history: list[dict[str, object]] = []
    previous_epoch: Optional[int] = None
    for history_index, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"Validation history entry {history_index} must be a mapping."
            )
        expected_history_fields = {
            "epoch",
            "step",
            "selection_metric_name",
            "selection_metric_value",
            "metrics",
        }
        if set(entry) != expected_history_fields:
            raise ValueError(
                f"Validation history entry {history_index} fields do not match "
                "the artifact schema."
            )
        if type(entry["epoch"]) is not int:
            raise TypeError(
                f"Validation history entry {history_index} epoch must be an integer."
            )
        entry_epoch = entry["epoch"]
        if entry_epoch <= 0:
            raise ValueError(
                f"Validation history entry {history_index} epoch must be positive."
            )
        entry_step = entry["step"]
        if type(entry_step) is not int:
            raise TypeError(
                f"Validation history entry {history_index} step must be an integer."
            )
        if entry_step < 0:
            raise ValueError(
                f"Validation history entry {history_index} step must be non-negative."
            )
        if entry["selection_metric_name"] != "overall.grounded_stage2_acc":
            raise ValueError(
                f"Validation history entry {history_index} "
                "selection_metric_name is invalid."
            )
        entry_metric_counts = validation_metrics_to_count_tensor(
            entry["metrics"],
            device=torch.device("cpu"),
        )
        entry_metrics = validation_count_tensor_to_metrics(entry_metric_counts)
        entry_score = validation_selection_score(entry_metrics)
        entry_selection_value = entry["selection_metric_value"]
        if type(entry_selection_value) not in (int, float):
            raise TypeError(
                f"Validation history entry {history_index} "
                "selection_metric_value must be a built-in number."
            )
        if (
            not math.isfinite(entry_selection_value)
            or float(entry_selection_value) != entry_score
        ):
            raise ValueError(
                f"Validation history entry {history_index} "
                "selection_metric_value is inconsistent with its metrics."
            )
        if previous_epoch is not None and entry_epoch <= previous_epoch:
            raise ValueError("Validation history epochs must be strictly increasing.")
        previous_epoch = entry_epoch
        canonical_history.append(
            {
                "epoch": entry_epoch,
                "step": entry_step,
                "selection_metric_name": "overall.grounded_stage2_acc",
                "selection_metric_value": entry_score,
                "metrics": entry_metrics,
            }
        )
    if previous_epoch is not None and epoch <= previous_epoch:
        raise ValueError(
            "Validation artifact epoch must be greater than the last history epoch."
        )
    score = validation_selection_score(metrics_snapshot)
    save_best = should_save_best(metrics_snapshot, best_score)
    normalized_best_score = None if best_score is None else float(best_score)
    candidate_history = canonical_history
    candidate_history.append(
        {
            "epoch": epoch,
            "step": step,
            "selection_metric_name": "overall.grounded_stage2_acc",
            "selection_metric_value": score,
            "metrics": copy.deepcopy(metrics_snapshot),
        }
    )
    output_dir = Path(output_dir)
    resulting_best_score = normalized_best_score
    if save_best:
        resulting_best_score = score
    epoch_path = output_dir / f"val_epoch_{epoch:03d}.json"
    history_path = output_dir / "validation_history.json"
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_paths: list[Path] = []
    try:
        file_descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{epoch_path.name}.", suffix=".stage", dir=output_dir
        )
        epoch_stage_path = Path(staging_name)
        staging_paths.append(epoch_stage_path)
        os.close(file_descriptor)
        file_descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{history_path.name}.", suffix=".stage", dir=output_dir
        )
        history_stage_path = Path(staging_name)
        staging_paths.append(history_stage_path)
        os.close(file_descriptor)
        file_descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{last_path.name}.", suffix=".stage", dir=output_dir
        )
        last_stage_path = Path(staging_name)
        staging_paths.append(last_stage_path)
        os.close(file_descriptor)
        best_stage_path: Optional[Path] = None
        if save_best:
            file_descriptor, staging_name = tempfile.mkstemp(
                prefix=f".{best_path.name}.", suffix=".stage", dir=output_dir
            )
            best_stage_path = Path(staging_name)
            staging_paths.append(best_stage_path)
            os.close(file_descriptor)

        dump_json(copy.deepcopy(metrics_snapshot), epoch_stage_path)
        dump_json(copy.deepcopy(candidate_history), history_stage_path)
        save_trainable_checkpoint(
            model,
            last_stage_path,
            args=args,
            epoch=epoch,
            step=step,
            extra_state={
                "selection_metric_name": "overall.grounded_stage2_acc",
                "selection_metric_value": score,
                "best_metric_name": "overall.grounded_stage2_acc",
                "best_metric_value": resulting_best_score,
                "val_metrics": copy.deepcopy(metrics_snapshot),
            },
        )
        if best_stage_path is not None:
            save_trainable_checkpoint(
                model,
                best_stage_path,
                args=args,
                epoch=epoch,
                step=step,
                extra_state={
                    "best_metric_name": "overall.grounded_stage2_acc",
                    "best_metric_value": score,
                    "val_metrics": copy.deepcopy(metrics_snapshot),
                },
            )

        os.replace(epoch_stage_path, epoch_path)
        if best_stage_path is not None:
            os.replace(best_stage_path, best_path)
        os.replace(last_stage_path, last_path)
        os.replace(history_stage_path, history_path)
        history[:] = candidate_history
        return float(resulting_best_score), save_best
    finally:
        for staging_path in staging_paths:
            try:
                staging_path.unlink(missing_ok=True)
            except OSError:
                pass


def build_model(args: argparse.Namespace) -> SpatialQwen3OmniGenerativeModel:
    raw_qwen3_omni_root = args.qwen3_omni_root
    if isinstance(raw_qwen3_omni_root, str) and not raw_qwen3_omni_root.strip():
        raise ValueError(
            "qwen3_omni_root (--qwen3-omni-root) must not be empty or whitespace."
        )
    qwen3_omni_root = Path(raw_qwen3_omni_root).expanduser().resolve()
    if not qwen3_omni_root.is_dir():
        raise ValueError(
            "qwen3_omni_root (--qwen3-omni-root) must be an existing local "
            "directory, but got: "
            f"{qwen3_omni_root}"
        )
    return SpatialQwen3OmniGenerativeModel(
        qwen3_omni_root=str(qwen3_omni_root),
        text_dtype=torch.bfloat16,
        audio_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        enable_llm_lora=True,
        llm_lora_r=args.llm_lora_r,
        llm_lora_alpha=args.llm_lora_alpha,
        llm_lora_dropout=args.llm_lora_dropout,
        llm_lora_num_layers=args.llm_lora_num_layers,
    )


def _broadcast_best_score(
    best_score: Optional[float],
    *,
    is_distributed: bool,
    device: torch.device,
    main_process: bool,
) -> Optional[float]:
    normalized_score: Optional[float] = None
    if main_process and best_score is not None:
        if (
            isinstance(best_score, bool)
            or not isinstance(best_score, Real)
            or not math.isfinite(best_score)
        ):
            raise ValueError("best_score must be a finite real number or None.")
        normalized_score = float(best_score)
        if not 0.0 <= normalized_score <= 1.0:
            raise ValueError("best_score must be in [0, 1].")
    if not is_distributed:
        return normalized_score

    presence = torch.tensor(
        int(main_process and normalized_score is not None),
        dtype=torch.int32,
        device=device,
    )
    value = torch.tensor(
        0.0 if normalized_score is None else normalized_score,
        dtype=torch.float64,
        device=device,
    )
    dist.broadcast(presence, src=0)
    dist.broadcast(value, src=0)
    presence_flag = int(presence.item())
    if presence_flag not in (0, 1):
        raise ValueError("Broadcast best_score presence flag must be 0 or 1.")
    if presence_flag == 0:
        return None
    received_score = float(value.item())
    if not math.isfinite(received_score) or not 0.0 <= received_score <= 1.0:
        raise ValueError("Broadcast best_score must be finite and in [0, 1].")
    return received_score


def run_training(args: argparse.Namespace) -> None:
    active_error: Optional[BaseException] = None
    try:
        args = _normalize_runtime_args(args)
        device, local_rank = resolve_device(args)
        is_distributed = init_training_distributed(device, local_rank)

        main_process = False
        process_stage_error: Optional[BaseException] = None
        try:
            main_process = is_main_process()
            rank = get_rank()
            seed_everything(args.seed + rank)
        except BaseException as error:
            process_stage_error = error
        raise_on_rank_failure(
            process_stage_error,
            device=device,
            stage="process identity and seeding",
        )

        runtime_error: Optional[BaseException] = None
        output_dir: Optional[Path] = None
        try:
            validate_args(args)
            raw_qwen3_omni_root = args.qwen3_omni_root
            if raw_qwen3_omni_root is None or (
                isinstance(raw_qwen3_omni_root, str) and not raw_qwen3_omni_root.strip()
            ):
                raise ValueError(
                    "qwen3_omni_root (--qwen3-omni-root) must be explicitly set."
                )
            qwen3_omni_root = Path(raw_qwen3_omni_root).expanduser().resolve()
            if not qwen3_omni_root.is_dir():
                raise ValueError(
                    "qwen3_omni_root (--qwen3-omni-root) must be an existing "
                    f"local directory, but got: {qwen3_omni_root}"
                )
            args.qwen3_omni_root = str(qwen3_omni_root)
            args.train_json = str(ensure_foa_json(args.train_json, "--train-json"))
            args.val_json = str(ensure_foa_json(args.val_json, "--val-json"))
            if Path(args.train_json) == Path(args.val_json):
                raise ValueError(
                    "--train-json and --val-json must point to different files."
                )
            raw_output_dir = args.output_dir
            if raw_output_dir is None or (
                isinstance(raw_output_dir, str) and not raw_output_dir.strip()
            ):
                raise ValueError("--output-dir must be explicitly set.")
            output_dir = Path(raw_output_dir).expanduser().resolve()
            if is_distributed and args.write_missing_features:
                raise ValueError(
                    "--write-missing-features is not supported with distributed "
                    "training."
                )
        except BaseException as error:
            runtime_error = error
        raise_on_rank_failure(
            runtime_error,
            device=device,
            stage="runtime validation",
        )
        assert output_dir is not None

        output_setup_error: Optional[BaseException] = None
        if main_process:
            try:
                output_dir = ensure_dir(output_dir)
            except BaseException as error:
                output_setup_error = error
        synchronize_main_process_error(
            output_setup_error,
            is_distributed=is_distributed,
            device=device,
            main_process=main_process,
            stage="output setup",
        )

        train_dataset = None
        train_dataset_error: Optional[BaseException] = None
        try:
            train_dataset = build_dataset(args)
        except BaseException as error:
            train_dataset_error = error
        raise_on_rank_failure(
            train_dataset_error,
            device=device,
            stage="training dataset build and preflight",
        )
        assert train_dataset is not None

        train_loader = None
        train_sampler = None
        train_loader_error: Optional[BaseException] = None
        try:
            train_loader, train_sampler = build_loader(
                train_dataset,
                args,
                is_distributed,
            )
        except BaseException as error:
            train_loader_error = error
        raise_on_rank_failure(
            train_loader_error,
            device=device,
            stage="training loader build",
        )
        assert train_loader is not None
        assert train_sampler is not None

        val_dataset = None
        val_dataset_error: Optional[BaseException] = None
        try:
            val_dataset = build_validation_dataset(args)
        except BaseException as error:
            val_dataset_error = error
        raise_on_rank_failure(
            val_dataset_error,
            device=device,
            stage="validation dataset build and preflight",
        )
        assert val_dataset is not None

        val_loader = None
        val_loader_error: Optional[BaseException] = None
        try:
            val_loader = build_validation_loader(
                val_dataset,
                args,
                is_distributed,
            )
        except BaseException as error:
            val_loader_error = error
        raise_on_rank_failure(
            val_loader_error,
            device=device,
            stage="validation loader build",
        )
        assert val_loader is not None

        core_model: Optional[nn.Module] = None
        model_build_error: Optional[BaseException] = None
        try:
            core_model = build_model(args)
        except BaseException as error:
            model_build_error = error
        raise_on_rank_failure(
            model_build_error,
            device=device,
            stage="model build",
        )
        assert core_model is not None

        model_move_error: Optional[BaseException] = None
        try:
            core_model = core_model.to(device)
        except BaseException as error:
            model_move_error = error
        raise_on_rank_failure(
            model_move_error,
            device=device,
            stage="model device move",
        )

        checkpoint_error: Optional[BaseException] = None
        if args.init_checkpoint:
            try:
                load_initial_checkpoint(core_model, args.init_checkpoint)
            except BaseException as error:
                checkpoint_error = error
        raise_on_rank_failure(
            checkpoint_error,
            device=device,
            stage="initial checkpoint load",
        )

        training_adapter: Optional[nn.Module] = None
        adapter_error: Optional[BaseException] = None
        try:
            training_adapter = ForwardTrainAdapter(core_model)
        except BaseException as error:
            adapter_error = error
        raise_on_rank_failure(
            adapter_error,
            device=device,
            stage="training adapter construction",
        )
        assert training_adapter is not None

        ddp_configuration_error: Optional[BaseException] = None
        try:
            if is_distributed and device.type == "cuda" and local_rank is None:
                raise ValueError("local_rank is required for CUDA DDP.")
        except BaseException as error:
            ddp_configuration_error = error
        raise_on_rank_failure(
            ddp_configuration_error,
            device=device,
            stage="DDP configuration validation",
        )
        training_model = wrap_ddp(
            training_adapter,
            is_distributed=is_distributed,
            device=device,
            local_rank=local_rank,
        )

        trainable_parameters: Optional[list[nn.Parameter]] = None
        training_optimizer: Optional[torch.optim.Optimizer] = None
        optimizer_error: Optional[BaseException] = None
        try:
            trainable_parameters = get_trainable_parameters(training_model)
            training_optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        except BaseException as error:
            optimizer_error = error
        raise_on_rank_failure(
            optimizer_error,
            device=device,
            stage="optimizer setup",
        )
        assert trainable_parameters is not None
        assert training_optimizer is not None

        global_step = 0
        best_score: Optional[float] = None
        validation_history: list[dict[str, object]] = []
        for epoch in range(1, args.epochs + 1):
            sampler_error: Optional[BaseException] = None
            try:
                if is_distributed:
                    train_sampler.set_epoch(epoch)
            except BaseException as error:
                sampler_error = error
            raise_on_rank_failure(
                sampler_error,
                device=device,
                stage="training sampler epoch setup",
            )

            summary, optimizer_steps = train_one_epoch(
                training_model,
                train_loader,
                training_optimizer,
                trainable_parameters,
                device,
                epoch=epoch,
                total_epochs=args.epochs,
                grad_accum=args.grad_accum,
                show_progress=main_process,
            )
            step_count_error: Optional[BaseException] = None
            try:
                if type(optimizer_steps) is not int or optimizer_steps < 0:
                    raise ValueError(
                        "train_one_epoch optimizer step count must be a "
                        "non-negative integer."
                    )
            except BaseException as error:
                step_count_error = error
            raise_on_rank_failure(
                step_count_error,
                device=device,
                stage="training step count validation",
            )
            global_step += optimizer_steps

            summary_error: Optional[BaseException] = None
            if main_process:
                try:
                    print(
                        f"epoch={epoch}/{args.epochs} step={global_step} "
                        f"train={summary}"
                    )
                except BaseException as error:
                    summary_error = error
            synchronize_main_process_error(
                summary_error,
                is_distributed=is_distributed,
                device=device,
                main_process=main_process,
                stage="training summary",
            )

            synchronize_trainable_buffers_for_validation(
                core_model,
                device=device,
            )

            local_metrics: Optional[Mapping[str, object]] = None
            evaluation_error: Optional[BaseException] = None
            try:
                local_metrics = evaluate_validation_epoch(
                    core_model,
                    val_loader,
                    device,
                    args=args,
                    main_process=main_process,
                )
            except BaseException as error:
                evaluation_error = error
            raise_on_rank_failure(
                evaluation_error,
                device=device,
                stage="validation evaluation",
            )
            assert local_metrics is not None
            metrics = reduce_validation_metrics(local_metrics, device=device)

            artifact_error: Optional[BaseException] = None
            if main_process:
                try:
                    current_score = validation_selection_score(metrics)
                    retained_score = (
                        current_score
                        if should_save_best(metrics, best_score)
                        else best_score
                    )
                    if args.save_every_epoch:
                        save_trainable_checkpoint(
                            core_model,
                            output_dir / f"epoch_{epoch:03d}.pt",
                            args=args,
                            epoch=epoch,
                            step=global_step,
                            extra_state={
                                "selection_metric_name": ("overall.grounded_stage2_acc"),
                                "selection_metric_value": current_score,
                                "best_metric_name": "overall.grounded_stage2_acc",
                                "best_metric_value": retained_score,
                                "val_metrics": copy.deepcopy(metrics),
                            },
                        )
                    best_score, saved_best = save_validation_artifacts(
                        model=core_model,
                        output_dir=output_dir,
                        args=args,
                        epoch=epoch,
                        step=global_step,
                        metrics=metrics,
                        history=validation_history,
                        best_score=best_score,
                    )
                    if type(saved_best) is not bool:
                        raise TypeError(
                            "save_validation_artifacts saved flag must be a bool."
                        )
                    if best_score is not None:
                        if (
                            isinstance(best_score, bool)
                            or not isinstance(best_score, Real)
                            or not math.isfinite(best_score)
                        ):
                            raise ValueError(
                                "save_validation_artifacts best_score must be a "
                                "finite real number or None."
                            )
                        best_score = float(best_score)
                        if not 0.0 <= best_score <= 1.0:
                            raise ValueError(
                                "save_validation_artifacts best_score must be in "
                                "[0, 1]."
                            )
                    print(
                        f"validation_epoch={epoch} "
                        f"grounded_stage2_acc={current_score:.6f} "
                        f"best={best_score:.6f} saved_best={saved_best}"
                    )
                except BaseException as error:
                    artifact_error = error
            synchronize_main_process_error(
                artifact_error,
                is_distributed=is_distributed,
                device=device,
                main_process=main_process,
                stage="validation artifact writing",
            )
            best_score = _broadcast_best_score(
                best_score if main_process else None,
                is_distributed=is_distributed,
                device=device,
                main_process=main_process,
            )
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            cleanup_distributed()
        except BaseException as cleanup_error:
            if active_error is None:
                raise
            try:
                cleanup_error_message = (
                    "cleanup_distributed failed while preserving the primary "
                    f"exception: {type(cleanup_error).__name__}: {cleanup_error}"
                )
            except BaseException:
                cleanup_error_message = (
                    "cleanup_distributed failed while preserving the primary "
                    f"exception: {type(cleanup_error).__name__}"
                )
            try:
                setattr(active_error, "cleanup_error", cleanup_error)
            except BaseException:
                pass
            try:
                setattr(
                    active_error,
                    "cleanup_error_message",
                    cleanup_error_message,
                )
            except BaseException:
                pass
            try:
                add_note = getattr(active_error, "add_note", None)
            except BaseException:
                add_note = None
            try:
                if callable(add_note):
                    add_note(cleanup_error_message)
            except BaseException:
                pass


def main(argv: Optional[Sequence[str]] = None) -> None:
    return run_training(parse_args(argv))


if __name__ == "__main__":
    main()
