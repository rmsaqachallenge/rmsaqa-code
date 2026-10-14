#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from tqdm import tqdm

from spatial_af3.layered_dataset import SpatialQAPairDataset, qa_pair_collate_fn
from spatial_af3.layered_evaluation import evaluate_spatial_model, validate_generated_batch
from spatial_af3.metrics import STAGE2_CATEGORIES
from spatial_af3.models.spatial_afnext_generative import SpatialAFNextGenerativeModel
from spatial_af3.utils import (
    cleanup_distributed,
    dump_json,
    ensure_dir,
    get_rank,
    is_main_process,
    seed_everything,
)

MAX_GRAD_NORM = 5.0


def ensure_foa_json(json_path: str | Path, arg_name: str) -> Path:
    del arg_name
    return Path(json_path).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Spatial AF3-Next concat-fusion model."
    )
    parser.add_argument("--afnext-root", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--val-feature-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-num-workers", type=int, default=None)
    parser.add_argument("--val-max-records", type=int, default=None)
    parser.add_argument("--val-max-new-tokens", type=int, default=48)
    parser.add_argument("--val-num-beams", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--extract-features-on-the-fly", action="store_true")
    parser.add_argument("--write-missing-features", action="store_true")
    parser.add_argument(
        "--audio-token-stride",
        type=int,
        default=3,
        help=(
            "Spatial preprocessing stride: cached Spatial frames are 1500 // "
            "stride. Default 3 gives 500 frames before fusion aligns them to "
            "the AF3-Next semantic width."
        ),
    )
    parser.add_argument("--llm-lora-r", type=int, default=16)
    parser.add_argument("--llm-lora-alpha", type=int, default=16)
    parser.add_argument("--llm-lora-dropout", type=float, default=0.05)
    parser.add_argument("--llm-lora-num-layers", type=int, default=28)
    parser.add_argument("--stage1-epochs", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
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
    if not 1 <= args.audio_token_stride <= 1500:
        raise ValueError("--audio-token-stride must be between 1 and 1500.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.val_num_workers < 0:
        raise ValueError("--val-num-workers must be non-negative.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and greater than zero.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative.")
    if not 0 <= args.llm_lora_dropout < 1:
        raise ValueError("--llm-lora-dropout must be in [0, 1).")
    if args.stage1_epochs < 0:
        raise ValueError("--stage1-epochs must be non-negative.")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be greater than zero when provided.")
    if args.val_max_records is not None and args.val_max_records <= 0:
        raise ValueError("--val-max-records must be greater than zero when provided.")
    if args.write_missing_features and not args.extract_features_on_the_fly:
        raise ValueError(
            "--write-missing-features requires --extract-features-on-the-fly."
        )


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
    requested_device = torch.device(args.device)
    if requested_device.type == "cpu":
        return torch.device("cpu"), None
    if requested_device.type != "cuda":
        return requested_device, None
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    environment_local_rank = os.environ.get("LOCAL_RANK")
    if environment_local_rank is not None:
        local_rank = int(environment_local_rank)
    elif requested_device.index is not None:
        local_rank = requested_device.index
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
            raise error
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
        raise RuntimeError(f"{stage} failed on rank {get_rank()}.") from local_error
    raise RuntimeError(f"{stage} failed on another rank.")


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
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas).")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

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


def build_model(args: argparse.Namespace) -> SpatialAFNextGenerativeModel:
    return SpatialAFNextGenerativeModel(
        afnext_root=args.afnext_root,
        enable_llm_lora=True,
        llm_lora_r=args.llm_lora_r,
        llm_lora_alpha=args.llm_lora_alpha,
        llm_lora_dropout=args.llm_lora_dropout,
        llm_lora_num_layers=args.llm_lora_num_layers,
    )


class ForwardTrainAdapter(nn.Module):
    def __init__(self, training_model: nn.Module) -> None:
        super().__init__()
        self.training_model = training_model

    def forward(self, *args, **kwargs):
        return self.training_model.forward_train(*args, **kwargs)


def call_forward_train(
    model: nn.Module,
    batch,
    device: torch.device,
    enable_layer2: bool,
):
    return model(batch, device=device, enable_layer2=enable_layer2)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: Sequence[nn.Parameter],
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
    enable_layer2: bool,
    show_progress: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stage = "layer1+2" if enable_layer2 else "layer1"
    progress = tqdm(
        loader,
        disable=not show_progress,
        desc=f"epoch {epoch}/{total_epochs} [{stage}]",
    )
    totals: dict[str, float] = {}
    batch_count = 0
    partial_count = 0
    optimizer_steps = 0
    is_distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )

    for batch in progress:
        outputs = call_forward_train(
            model,
            batch,
            device=device,
            enable_layer2=enable_layer2,
        )
        loss = outputs.get("loss")
        loss_is_scalar = (
            torch.is_tensor(loss)
            and loss.numel() == 1
            and not loss.is_complex()
        )
        loss_is_finite = (
            loss_is_scalar
            and torch.isfinite(loss.detach()).all().item()
        )
        metrics_are_finite = True
        output_scalars: dict[str, float] = {}
        for key, value in outputs.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            if value.is_complex() or not torch.isfinite(value.detach()).all().item():
                metrics_are_finite = False
                continue
            output_scalars[key] = float(value.detach().item())

        local_valid = bool(loss_is_finite and metrics_are_finite)
        validity = torch.tensor(
            int(local_valid),
            dtype=torch.int32,
            device=device,
        )
        if is_distributed:
            torch.distributed.all_reduce(
                validity,
                op=torch.distributed.ReduceOp.MIN,
            )
        if validity.item() == 0:
            raise FloatingPointError(
                "At least one rank produced an invalid training loss or metric."
            )

        for key, value in output_scalars.items():
            totals[key] = totals.get(key, 0.0) + value

        (loss / grad_accum).backward()
        batch_count += 1
        partial_count += 1

        progress.set_postfix(
            loss=output_scalars["loss"],
            layer1=output_scalars.get("loss_layer1_next_token", math.nan),
            layer2=output_scalars.get("loss_layer2_next_token", math.nan),
        )

        if partial_count == grad_accum:
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=MAX_GRAD_NORM,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            partial_count = 0

    if batch_count == 0:
        raise ValueError("Training loader is empty because it yielded no batches.")

    if partial_count > 0:
        correction = grad_accum / partial_count
        for parameter in trainable_parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=MAX_GRAD_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    metric_names = sorted(totals)
    reduced_values = torch.tensor(
        [*(totals[name] for name in metric_names), float(batch_count)],
        dtype=torch.float64,
        device=device,
    )
    if is_distributed:
        torch.distributed.all_reduce(
            reduced_values,
            op=torch.distributed.ReduceOp.SUM,
        )
    if not torch.isfinite(reduced_values).all().item():
        raise FloatingPointError("Reduced training metrics must be finite.")
    global_batch_count = float(reduced_values[-1].item())
    if global_batch_count <= 0:
        raise ValueError("Global training batch count must be greater than zero.")
    summary = {
        name: float(reduced_values[index].item()) / global_batch_count
        for index, name in enumerate(metric_names)
    }
    return summary, optimizer_steps


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DDP):
        model = model.module
    if isinstance(model, ForwardTrainAdapter):
        model = model.training_model
    return model


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters.")
    return parameters


def load_initial_checkpoint(model: nn.Module, checkpoint_path: str | Path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a Mapping.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a Mapping.")
    return unwrap_model(model).load_state_dict(state_dict, strict=False)


def save_trainable_checkpoint(
    model: nn.Module,
    out_path: Path,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    extra_state: Optional[dict[str, object]] = None,
) -> None:
    core_model = unwrap_model(model)
    trainable_names = {
        name for name, parameter in core_model.named_parameters() if parameter.requires_grad
    }
    core_state = core_model.state_dict()
    missing_trainable_names = trainable_names - set(core_state)
    if missing_trainable_names:
        raise ValueError(
            "Model trainable parameters are missing from state_dict: "
            + ", ".join(sorted(missing_trainable_names))
        )
    related_buffer_names = set()
    for module_name, module in core_model.named_modules():
        if any(
            parameter.requires_grad
            for parameter in module.parameters(recurse=False)
        ):
            for buffer_name, _ in module.named_buffers(recurse=False):
                qualified_name = (
                    f"{module_name}.{buffer_name}" if module_name else buffer_name
                )
                if qualified_name in core_state:
                    related_buffer_names.add(qualified_name)
    checkpoint_names = trainable_names | related_buffer_names
    state = {
        "state_dict": {
            name: value.cpu()
            for name, value in core_state.items()
            if name in checkpoint_names
        },
        "args": vars(args).copy(),
        "epoch": epoch,
        "step": step,
    }
    if extra_state:
        overlapping_keys = set(state).intersection(extra_state)
        if overlapping_keys:
            raise ValueError(
                "extra_state cannot overwrite checkpoint keys: "
                f"{sorted(overlapping_keys)}"
            )
        state.update(extra_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)


def _count_from_accuracy(
    metric: dict[str, object],
    key: str,
    num_records: int,
) -> int:
    if num_records == 0:
        return 0
    return int(round(float(metric[key]) * num_records))


def _metric_to_counts(metric):
    n = int(metric["num_records"])
    return [n, round(metric["stage1_acc"] * n), round(metric["stage2_acc_raw"] * n), round(metric["grounded_stage2_acc"] * n)]



def validation_metrics_to_count_tensor(
    metrics: dict[str, object],
    *,
    device: torch.device,
) -> torch.Tensor:
    counts = _metric_to_counts(metrics["overall"])
    categories = metrics["categories"]
    for category_id in STAGE2_CATEGORIES:
        counts.extend(_metric_to_counts(categories[str(category_id)]))
    return torch.tensor(counts, dtype=torch.float64, device=device)


def _counts_to_metric(counts):
    n, stage1, stage2_raw, grounded = counts
    denom = max(n, 1)
    return {"num_records": n, "stage1_acc": stage1 / denom, "stage2_acc_raw": stage2_raw / denom, "grounded_stage2_acc": grounded / denom}



def validation_count_tensor_to_metrics(
    count_tensor: torch.Tensor,
) -> dict[str, object]:
    values = [
        int(round(float(value)))
        for value in count_tensor.detach().cpu().tolist()
    ]
    expected = 4 * (1 + len(STAGE2_CATEGORIES))
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} validation count values, got {len(values)}."
        )
    offset = 0
    overall = _counts_to_metric(values[offset : offset + 4])
    offset += 4
    categories: dict[str, object] = {}
    for category_id, category_name in STAGE2_CATEGORIES.items():
        categories[str(category_id)] = {
            "category_id": category_id,
            "category_name": category_name,
            **_counts_to_metric(values[offset : offset + 4]),
        }
        offset += 4
    return {"overall": overall, "categories": categories}


def reduce_validation_metrics(
    metrics: dict[str, object],
    *,
    device: torch.device,
) -> dict[str, object]:
    counts = validation_metrics_to_count_tensor(metrics, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return validation_count_tensor_to_metrics(counts)


def validation_selection_score(metrics: dict[str, object]) -> float:
    overall = metrics["overall"]
    if not isinstance(overall, dict):
        raise TypeError("Validation metrics 'overall' entry must be a dictionary.")
    return float(overall["grounded_stage2_acc"])


def should_save_best(
    metrics: dict[str, object],
    best_score: float,
    enable_layer2: bool,
) -> bool:
    return enable_layer2 and validation_selection_score(metrics) > best_score


def save_validation_artifacts(
    *,
    model: nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    metrics: dict[str, object],
    enable_layer2: bool,
    best_score: float,
    validation_history: list[dict[str, object]],
) -> float:
    score = validation_selection_score(metrics)
    dump_json(metrics, output_dir / f"val_epoch_{epoch:03d}.json")
    validation_history.append(
        {
            "epoch": epoch,
            "step": step,
            "enable_layer2": enable_layer2,
            "selection_metric_name": "overall.grounded_stage2_acc",
            "selection_metric_value": score,
            "metrics": metrics,
        }
    )
    dump_json(validation_history, output_dir / "validation_history.json")
    save_trainable_checkpoint(
        model,
        output_dir / "last.pt",
        args=args,
        epoch=epoch,
        step=step,
    )
    if should_save_best(
        metrics,
        best_score=best_score,
        enable_layer2=enable_layer2,
    ):
        best_score = score
        save_trainable_checkpoint(
            model,
            output_dir / "best.pt",
            args=args,
            epoch=epoch,
            step=step,
            extra_state={
                "best_metric_name": "overall.grounded_stage2_acc",
                "best_metric_value": score,
                "val_metrics": metrics,
            },
        )
    return best_score


def run_training(args: argparse.Namespace) -> None:
    try:
        device, local_rank = resolve_device(args)
        is_distributed = init_training_distributed(device, local_rank)
        args.train_json = str(ensure_foa_json(args.train_json, "--train-json"))
        args.val_json = str(ensure_foa_json(args.val_json, "--val-json"))
        if Path(args.train_json) == Path(args.val_json):
            raise ValueError(
                "--train-json and --val-json must point to different files."
            )
        if is_distributed and args.write_missing_features:
            raise ValueError(
                "--write-missing-features is not supported with distributed training."
            )
        seed_everything(args.seed + get_rank())

        main_process = is_main_process()
        output_setup_error: Optional[BaseException] = None
        if main_process:
            try:
                output_dir = ensure_dir(args.output_dir)
                dump_json(vars(args), output_dir / "train_args.json")
            except BaseException as exc:
                output_setup_error = exc
                output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.output_dir)
        synchronize_main_process_error(
            output_setup_error,
            is_distributed=is_distributed,
            device=device,
            main_process=main_process,
            stage="output setup",
        )

        dataset = build_dataset(args)
        loader, sampler = build_loader(dataset, args, is_distributed)
        validation_setup_error: Optional[BaseException] = None
        try:
            val_dataset = build_validation_dataset(args)
            val_loader = build_validation_loader(
                val_dataset,
                args,
                is_distributed,
            )
        except BaseException as exc:
            validation_setup_error = exc
        raise_on_rank_failure(
            validation_setup_error,
            device=device,
            stage="validation setup",
        )

        core_model = build_model(args).to(device)
        if args.init_checkpoint:
            load_result = load_initial_checkpoint(core_model, args.init_checkpoint)
            if main_process:
                print(
                    f"loaded_init_checkpoint={args.init_checkpoint} "
                    f"missing={len(load_result.missing_keys)} "
                    f"unexpected={len(load_result.unexpected_keys)}"
                )

        model = ForwardTrainAdapter(core_model)
        model = wrap_ddp(
            model,
            is_distributed=is_distributed,
            device=device,
            local_rank=local_rank,
        )
        trainable_parameters = get_trainable_parameters(model)
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        global_step = 0
        best_score = float("-inf")
        validation_history: list[dict[str, object]] = []
        for epoch in range(1, args.epochs + 1):
            if is_distributed:
                sampler.set_epoch(epoch)
            enable_layer2 = epoch > args.stage1_epochs
            summary, optimizer_steps = train_one_epoch(
                model,
                loader,
                optimizer,
                trainable_parameters,
                device,
                epoch=epoch,
                total_epochs=args.epochs,
                grad_accum=args.grad_accum,
                enable_layer2=enable_layer2,
                show_progress=main_process,
            )
            global_step += optimizer_steps

            if is_distributed:
                dist.barrier()
            summary_error: Optional[BaseException] = None
            if main_process:
                try:
                    dump_json(
                        summary,
                        output_dir / f"train_epoch_{epoch:03d}.json",
                    )
                except BaseException as exc:
                    summary_error = exc
            synchronize_main_process_error(
                summary_error,
                is_distributed=is_distributed,
                device=device,
                main_process=main_process,
                stage=f"epoch {epoch} training summary",
            )

            local_val_metrics: Optional[dict[str, object]] = None
            validation_error: Optional[BaseException] = None
            try:
                local_val_metrics, _ = evaluate_spatial_model(
                    unwrap_model(model),
                    val_loader,
                    device,
                    max_new_tokens=args.val_max_new_tokens,
                    num_beams=args.val_num_beams,
                    desc=f"val {epoch}/{args.epochs}",
                    collect_predictions=False,
                    show_progress=main_process,
                    validate_batch=validate_generated_batch,
                )
            except BaseException as exc:
                validation_error = exc
            raise_on_rank_failure(
                validation_error,
                device=device,
                stage=f"epoch {epoch} validation",
            )
            assert local_val_metrics is not None
            val_metrics = reduce_validation_metrics(
                local_val_metrics,
                device=device,
            )

            artifact_error: Optional[BaseException] = None
            if main_process:
                try:
                    best_score = save_validation_artifacts(
                        model=model,
                        output_dir=output_dir,
                        args=args,
                        epoch=epoch,
                        step=global_step,
                        metrics=val_metrics,
                        enable_layer2=enable_layer2,
                        best_score=best_score,
                        validation_history=validation_history,
                    )
                    if args.save_every_epoch:
                        save_trainable_checkpoint(
                            model,
                            output_dir / f"epoch_{epoch:03d}.pt",
                            args=args,
                            epoch=epoch,
                            step=global_step,
                        )
                    current_score = validation_selection_score(val_metrics)
                    print(
                        f"val_overall_grounded_stage2_acc={current_score:.6f} "
                        f"best={best_score:.6f}"
                    )
                except BaseException as exc:
                    artifact_error = exc
            synchronize_main_process_error(
                artifact_error,
                is_distributed=is_distributed,
                device=device,
                main_process=main_process,
                stage=f"epoch {epoch} validation artifact writing",
            )
            if is_distributed:
                best_score_tensor = torch.tensor(
                    best_score,
                    dtype=torch.float64,
                    device=device,
                )
                dist.broadcast(best_score_tensor, src=0)
                best_score = float(best_score_tensor.item())
                dist.barrier()
    finally:
        cleanup_distributed()


def main(argv: Optional[Sequence[str]] = None):
    return run_training(parse_args(argv))


if __name__ == "__main__":
    main()
