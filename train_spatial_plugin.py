#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from tqdm import tqdm

from spatial_af3.dataset import SpatialQAPairDataset, qa_pair_collate_fn
from spatial_af3.evaluation import evaluate_spatial_model
from spatial_af3.metrics import STAGE2_CATEGORIES
from spatial_af3.models import SpatialAF3GenerativeModel
from spatial_af3.utils import (
    TRAINABLE_CHECKPOINT_FORMAT_VERSION,
    cleanup_distributed,
    dump_json,
    ensure_dir,
    get_rank,
    init_distributed,
    is_main_process,
    load_trainable_state_dict,
    seed_everything,
    trainable_state_names,
)

DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_JSON = DATA_ROOT / "train_QA"
DEFAULT_VAL_JSON = DATA_ROOT / "dev_QA"


def ensure_foa_json(json_path: str | Path, arg_name: str) -> Path:
    path = Path(json_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{arg_name} path does not exist: {path}")
    if path.is_file() and path.suffix.lower() == ".json":
        return path
    if not path.is_dir():
        raise ValueError(f"{arg_name} must point to a JSON file or a directory: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-AF3 spatial sidecar.")
    parser.add_argument("--af3-root", required=True)
    parser.add_argument(
        "--train-json",
        default=str(DEFAULT_TRAIN_JSON),
        help="Training split as a legacy aggregate JSON file or a per-sample JSON directory.",
    )
    parser.add_argument(
        "--val-json",
        default=str(DEFAULT_VAL_JSON),
        help="Validation split as a legacy aggregate JSON file or a per-sample JSON directory.",
    )
    parser.add_argument("--feature-root", required=True)
    parser.add_argument(
        "--val-feature-root",
        default=None,
        help="Validation feature directory. Defaults to --feature-root.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Validation batch size. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=None,
        help="Validation worker count. Defaults to --num-workers.",
    )
    parser.add_argument("--val-max-records", type=int, default=None)
    parser.add_argument("--val-max-new-tokens", type=int, default=48)
    parser.add_argument("--val-num-beams", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--task-id", type=int, default=None, help="Filter dataset to specific task_id (1=QA, 2=Action).")
    parser.add_argument("--init-checkpoint", type=str, default="", help="Optional trainable checkpoint to initialize the spatial plugin before training.")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--extract-features-on-the-fly", action="store_true")
    parser.add_argument("--write-missing-features", action="store_true")
    parser.add_argument("--audio-token-stride", type=int, default=3, help="Spatial feature resampling stride used during preprocessing/loading. Keep 3 so spatial tokens stay at 500 and match the truncated AF-Whisper branch.")
    parser.add_argument("--disable-stage35", action="store_true", help="Do not load AF3 stage35 LoRA/chat adapter.")
    parser.add_argument("--disable-llm-lora", action="store_true", help="Disable trainable LLM LoRA adapters.")
    parser.add_argument("--llm-lora-r", type=int, default=16, help="LoRA rank for AF3/Qwen2 LLM adapters.")
    parser.add_argument("--llm-lora-alpha", type=int, default=16, help="LoRA alpha for AF3/Qwen2 LLM adapters.")
    parser.add_argument("--llm-lora-dropout", type=float, default=0.05, help="LoRA dropout for AF3/Qwen2 LLM adapters.")
    parser.add_argument("--llm-lora-num-layers", type=int, default=28, help="Apply LLM LoRA to the first N transformer layers. Default 28 = all layers.")
    parser.add_argument("--stage1-epochs", type=int, default=0, help="Number of epochs to train on stage1 only before enabling stage2.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.val_feature_root = args.val_feature_root or args.feature_root
    args.val_batch_size = args.val_batch_size or args.batch_size
    args.val_num_workers = (
        args.num_workers if args.val_num_workers is None else args.val_num_workers
    )
    return args


class ForwardTrainAdapter(torch.nn.Module):
    def __init__(self, training_model: torch.nn.Module) -> None:
        super().__init__()
        self.training_model = training_model

    def forward(self, *args, **kwargs):
        return self.training_model.forward_train(*args, **kwargs)


def call_forward_train(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    enable_stage2: bool,
):
    return model(batch, device=device, enable_stage2=enable_stage2)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        model = model.module
    if isinstance(model, ForwardTrainAdapter):
        model = model.training_model
    return model


def validate_training_outputs(
    outputs: Dict[str, torch.Tensor], *, device: torch.device
) -> Dict[str, float]:
    loss = outputs.get("loss")
    loss_is_scalar = (
        torch.is_tensor(loss)
        and loss.numel() == 1
        and not loss.is_complex()
        and loss.requires_grad
        and loss.grad_fn is not None
    )
    local_valid = bool(
        loss_is_scalar and torch.isfinite(loss.detach()).all().item()
    )
    output_scalars: Dict[str, float] = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value) or value.numel() != 1:
            continue
        if value.is_complex() or not torch.isfinite(value.detach()).all().item():
            local_valid = False
            continue
        output_scalars[key] = float(value.detach().item())

    validity = torch.tensor(int(local_valid), dtype=torch.int32, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(validity, op=torch.distributed.ReduceOp.MIN)
    if validity.item() == 0:
        raise FloatingPointError(
            "At least one rank produced an invalid training loss or metric."
        )
    return output_scalars


def synchronize_rank_success(local_success: bool, device: torch.device) -> bool:
    success = torch.tensor(int(local_success), dtype=torch.int32, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(success, op=torch.distributed.ReduceOp.MIN)
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


def log_rank_phase(message: str) -> None:
    print(f"[rank {get_rank()}] {message}", flush=True)


def save_trainable_checkpoint(
    model: torch.nn.Module,
    out_path: Path,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    extra_state: Optional[Dict[str, object]] = None,
) -> None:
    model_to_save = unwrap_model(model)
    state_names = trainable_state_names(model_to_save)
    state = {
        "checkpoint_format_version": TRAINABLE_CHECKPOINT_FORMAT_VERSION,
        "state_dict": {
            k: v.cpu()
            for k, v in model_to_save.state_dict().items()
            if k in state_names
        },
        "args": vars(args),
        "epoch": epoch,
        "step": step,
    }
    if extra_state:
        state.update(extra_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)


def validation_selection_score(metrics: Dict[str, object]) -> float:
    return float(metrics["overall"]["grounded_stage2_acc"])


def should_save_best(
    metrics: Dict[str, object], best_score: float, enable_stage2: bool
) -> bool:
    return enable_stage2 and validation_selection_score(metrics) > best_score


class DistributedEvalSampler(Sampler[int]):
    """Shard validation indices across ranks without padding duplicate samples."""

    def __init__(
        self,
        dataset,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                raise ValueError(
                    "num_replicas is required when distributed is not initialized."
                )
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                raise ValueError(
                    "rank is required when distributed is not initialized."
                )
            rank = torch.distributed.get_rank()
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


def _counts_to_list(counts: Dict[str, int]) -> List[int]:
    return [
        int(counts["num_records"]),
        int(counts["stage1_correct"]),
        int(counts["stage2_correct_raw"]),
        int(counts["grounded_stage2_correct"]),
    ]


def validation_counts_to_count_tensor(
    counts: Dict[str, object], *, device: torch.device
) -> torch.Tensor:
    values: List[int] = []
    values.extend(_counts_to_list(counts["overall"]))
    categories = counts["categories"]
    for category_id in STAGE2_CATEGORIES:
        values.extend(_counts_to_list(categories[str(category_id)]))
    return torch.tensor(values, dtype=torch.float64, device=device)


def _counts_to_metric(counts: List[int]) -> Dict[str, float]:
    num_records, stage1_correct, stage2_correct_raw, grounded_stage2_correct = counts
    denom = max(num_records, 1)
    return {
        "stage1_acc": stage1_correct / denom,
        "stage2_acc_raw": stage2_correct_raw / denom,
        "grounded_stage2_acc": grounded_stage2_correct / denom,
    }


def validation_count_tensor_to_metrics(count_tensor: torch.Tensor) -> Dict[str, object]:
    values = [int(round(float(value))) for value in count_tensor.detach().cpu().tolist()]
    offset = 0
    overall = _counts_to_metric(values[offset : offset + 4])
    offset += 4
    categories: Dict[str, object] = {}
    for category_id, category_name in STAGE2_CATEGORIES.items():
        categories[str(category_id)] = {
            "category_id": category_id,
            "category_name": category_name,
            **_counts_to_metric(values[offset : offset + 4]),
        }
        offset += 4
    return {"overall": overall, "categories": categories}


def reduce_validation_counts(
    counts: Dict[str, object], *, device: torch.device
) -> Dict[str, object]:
    count_tensor = validation_counts_to_count_tensor(counts, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
    return validation_count_tensor_to_metrics(count_tensor)


def save_validation_artifacts(
    *,
    model: torch.nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    metrics: Dict[str, object],
    enable_stage2: bool,
    best_score: float,
    validation_history: List[Dict[str, object]],
) -> float:
    score = validation_selection_score(metrics)
    dump_json(metrics, output_dir / f"val_epoch_{epoch:03d}.json")
    validation_history.append(
        {
            "epoch": epoch,
            "step": step,
            "enable_stage2": enable_stage2,
            "selection_metric_name": "overall.grounded_stage2_acc",
            "selection_metric_value": score,
            "metrics": metrics,
        }
    )
    dump_json(validation_history, output_dir / "validation_history.json")
    save_trainable_checkpoint(
        model, output_dir / "last.pt", args=args, epoch=epoch, step=step
    )
    if should_save_best(metrics, best_score=best_score, enable_stage2=enable_stage2):
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


def main() -> None:
    args = parse_args()
    args.train_json = str(ensure_foa_json(args.train_json, "--train-json"))
    args.val_json = str(ensure_foa_json(args.val_json, "--val-json"))
    if Path(args.train_json) == Path(args.val_json):
        raise ValueError("--train-json and --val-json must point to different data paths.")
    is_distributed = init_distributed()
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else args.device)
    seed_everything(args.seed + get_rank())

    output_dir = Path(args.output_dir)
    output_setup_error: Optional[BaseException] = None
    if is_main_process():
        try:
            output_dir = ensure_dir(args.output_dir)
            dump_json(vars(args), output_dir / "train_args.json")
        except BaseException as exc:
            output_setup_error = exc
    raise_on_rank_failure(
        output_setup_error,
        device=device,
        stage="output setup",
    )
    log_rank_phase("phase=output_setup_complete")

    dataset = SpatialQAPairDataset(
        json_path=args.train_json,
        feature_root=args.feature_root,
        extract_features_on_the_fly=args.extract_features_on_the_fly,
        write_missing_features=args.write_missing_features,
        audio_token_stride=args.audio_token_stride,
        max_records=args.max_records,
        task_id=args.task_id,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else RandomSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=qa_pair_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_dataset = SpatialQAPairDataset(
        json_path=args.val_json,
        feature_root=args.val_feature_root,
        extract_features_on_the_fly=args.extract_features_on_the_fly,
        write_missing_features=args.write_missing_features,
        audio_token_stride=args.audio_token_stride,
        max_records=args.val_max_records,
        task_id=args.task_id,
    )
    val_sampler = (
        DistributedEvalSampler(val_dataset)
        if is_distributed
        else SequentialSampler(val_dataset)
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.val_batch_size,
        num_workers=args.val_num_workers,
        collate_fn=qa_pair_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    core_model = SpatialAF3GenerativeModel(
        af3_root=args.af3_root,
        apply_stage35=not args.disable_stage35,
        enable_llm_lora=not args.disable_llm_lora,
        llm_lora_r=args.llm_lora_r,
        llm_lora_alpha=args.llm_lora_alpha,
        llm_lora_dropout=args.llm_lora_dropout,
        llm_lora_num_layers=args.llm_lora_num_layers,
    ).to(device)
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        missing_nonrequired_keys, unexpected_keys = load_trainable_state_dict(
            core_model,
            init_ckpt["state_dict"],
            checkpoint_format_version=init_ckpt.get("checkpoint_format_version"),
        )
        if is_main_process():
            print(
                f"loaded_init_checkpoint={args.init_checkpoint} "
                f"missing_nonrequired_keys={len(missing_nonrequired_keys)} "
                f"unexpected_keys={len(unexpected_keys)}"
            )
    model = ForwardTrainAdapter(core_model)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    best_score = float("-inf")
    validation_history: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        log_rank_phase(f"epoch={epoch} phase=train_start")
        if is_distributed:
            sampler.set_epoch(epoch)
        model.train()
        epoch_stats = {}
        enable_stage2 = epoch > args.stage1_epochs
        pbar = tqdm(loader, disable=not is_main_process(), desc=f"epoch {epoch}/{args.epochs} {'[stage1]' if not enable_stage2 else '[stage1+2]'}")
        optimizer.zero_grad(set_to_none=True)
        last_step = 0
        for step, batch in enumerate(pbar, start=1):
            last_step = step
            outputs = call_forward_train(
                model,
                batch,
                device=device,
                enable_stage2=enable_stage2,
            )
            output_scalars = validate_training_outputs(outputs, device=device)
            loss = outputs["loss"] / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            for key, val in output_scalars.items():
                epoch_stats.setdefault(key, 0.0)
                epoch_stats[key] += val
            avg_stats = {k: v / step for k, v in epoch_stats.items() if k != "loss"}
            pbar.set_postfix(
                loss=output_scalars["loss"],
                gate2=avg_stats.get("mean_gate_stage2", math.nan),
            )

        log_rank_phase(f"epoch={epoch} phase=train_complete last_step={last_step}")

        summary_error: Optional[BaseException] = None
        if is_main_process():
            try:
                summary = {
                    k: v / max(len(loader), 1) for k, v in epoch_stats.items()
                }
                dump_json(summary, output_dir / f"train_epoch_{epoch:03d}.json")
            except BaseException as exc:
                summary_error = exc
        raise_on_rank_failure(
            summary_error,
            device=device,
            stage=f"epoch {epoch} training summary",
        )
        log_rank_phase(
            f"epoch={epoch} last_step={last_step} phase=validation_start"
        )
        local_val_counts = None
        validation_error: Optional[BaseException] = None
        try:
            _, _, local_val_counts = evaluate_spatial_model(
                unwrap_model(model),
                val_loader,
                device,
                max_new_tokens=args.val_max_new_tokens,
                num_beams=args.val_num_beams,
                desc=f"val {epoch}/{args.epochs}",
                collect_predictions=False,
                show_progress=is_main_process(),
                return_counts=True,
            )
        except BaseException as exc:
            validation_error = exc
        raise_on_rank_failure(
            validation_error,
            device=device,
            stage=f"epoch {epoch} validation",
        )
        log_rank_phase(
            f"epoch={epoch} last_step={last_step} phase=validation_complete"
        )
        val_metrics = reduce_validation_counts(local_val_counts, device=device)
        log_rank_phase(
            f"epoch={epoch} last_step={last_step} "
            "phase=metric_reduction_complete"
        )
        artifact_error: Optional[BaseException] = None
        if is_main_process():
            try:
                best_score = save_validation_artifacts(
                    model=model,
                    output_dir=output_dir,
                    args=args,
                    epoch=epoch,
                    step=global_step,
                    metrics=val_metrics,
                    enable_stage2=enable_stage2,
                    best_score=best_score,
                    validation_history=validation_history,
                )
                if args.save_every_epoch:
                    save_trainable_checkpoint(model, output_dir / f"epoch_{epoch:03d}.pt", args=args, epoch=epoch, step=global_step)
                current_score = validation_selection_score(val_metrics)
                print(
                    f"val_overall_grounded_stage2_acc={current_score:.6f} "
                    f"best={best_score:.6f}"
                )
            except BaseException as exc:
                artifact_error = exc
        raise_on_rank_failure(
            artifact_error,
            device=device,
            stage=f"epoch {epoch} validation artifact writing",
        )
        if is_distributed:
            best_score_tensor = torch.tensor(best_score, dtype=torch.float64, device=device)
            torch.distributed.broadcast(best_score_tensor, src=0)
            best_score = float(best_score_tensor.item())
            torch.distributed.barrier()
        log_rank_phase(
            f"epoch={epoch} last_step={last_step} phase=checkpoint_sync_complete"
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
