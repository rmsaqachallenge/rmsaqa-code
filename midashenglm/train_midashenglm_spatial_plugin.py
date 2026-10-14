import argparse
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm

from spatial_af3.midashenglm_dataset import (
    MiDashengSpatialQAPairDataset,
    collate_midasheng_spatial,
)
from spatial_af3.layered_mcq import extract_mcq_option
from spatial_af3.metrics import CategoryHierarchicalAccuracyMeter, STAGE2_CATEGORIES
from spatial_af3.models.midashenglm_spatial_training import SpatialMiDashengLMTrainingModel
from spatial_af3.utils import (
    cleanup_distributed,
    dump_json,
    init_distributed,
    is_main_process,
    seed_everything,
)


def _positive_int(value):
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive integer")
    if not text.lstrip("+").isdigit() or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value):
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    if not text.lstrip("+").isdigit() or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--midashenglm-root", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--val-feature-root")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--frontend-version", default="dasheng-64mel-v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=_positive_int, default=3)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--val-batch-size", type=_positive_int)
    parser.add_argument("--val-num-workers", type=_nonnegative_int)
    parser.add_argument("--val-max-records", type=_positive_int)
    parser.add_argument("--val-max-new-tokens", type=_positive_int, default=48)
    parser.add_argument("--val-num-beams", type=_positive_int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--disable-layer2", action="store_true")
    parser.add_argument("--disable-llm-lora", action="store_true")
    parser.add_argument("--llm-lora-r", type=_positive_int, default=16)
    parser.add_argument("--llm-lora-alpha", type=_positive_int, default=16)
    parser.add_argument("--llm-lora-dropout", type=float, default=0.05)
    parser.add_argument("--llm-lora-num-layers", type=_positive_int)
    parser.add_argument("--stage1-epochs", type=_nonnegative_int, default=0)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-records", type=_positive_int)
    args = parser.parse_args(argv)
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if not 0 <= args.llm_lora_dropout < 1:
        parser.error("--llm-lora-dropout must be in [0, 1)")
    if bool(args.val_json) != bool(args.val_feature_root):
        parser.error("--val-json and --val-feature-root must be provided together")
    args.val_batch_size = args.val_batch_size or args.batch_size
    args.val_num_workers = (
        args.num_workers if args.val_num_workers is None else args.val_num_workers
    )
    return args


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


class _DistributedEvalSampler(Sampler):
    """Shard validation data without padding duplicate samples."""

    def __init__(self, dataset):
        if not dist.is_available() or not dist.is_initialized():
            raise ValueError("distributed process group is not initialized")
        self.dataset = dataset
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        if len(self.dataset) <= self.rank:
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.num_replicas) + 1


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _make_loader(dataset, args, shuffle, distributed=False, train=False):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle) if train else _DistributedEvalSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.val_batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers if train else args.val_num_workers,
        collate_fn=collate_midasheng_spatial,
        pin_memory=True,
        drop_last=False,
    )


def _metric_to_counts(metric):
    n = int(metric["num_records"])
    return [n, round(metric["stage1_acc"] * n), round(metric["stage2_acc_raw"] * n), round(metric["grounded_stage2_acc"] * n)]



def _category_metrics_to_count_tensor(metrics, *, device):
    values = _metric_to_counts(metrics["overall"])
    for category_id in STAGE2_CATEGORIES:
        values.extend(_metric_to_counts(metrics["categories"][str(category_id)]))
    return torch.tensor(values, dtype=torch.float64, device=device)


def _counts_to_metric(values):
    n, stage1, stage2_raw, grounded = values
    denom = max(n, 1)
    return {"num_records": n, "stage1_acc": stage1 / denom, "stage2_acc_raw": stage2_raw / denom, "grounded_stage2_acc": grounded / denom}



def _category_count_tensor_to_metrics(count_tensor):
    values = [int(round(float(value))) for value in count_tensor.detach().cpu().tolist()]
    metrics = {"overall": _counts_to_metric(values[:4]), "categories": {}}
    offset = 4
    for category_id, category_name in STAGE2_CATEGORIES.items():
        metrics["categories"][str(category_id)] = {
            "category_id": category_id,
            "category_name": category_name,
            **_counts_to_metric(values[offset : offset + 4]),
        }
        offset += 4
    return metrics


@torch.no_grad()
def _evaluate_stage2_accuracy(
    model, loader, device, *, max_new_tokens=12, num_beams=1
):
    """Generate validation answers and return the hierarchical metrics."""
    model.eval()
    model_for_generation = _unwrap_model(model)
    meter = CategoryHierarchicalAccuracyMeter()
    for batch in loader:
        moved_batch = _move(batch, device)
        outputs = model_for_generation.generate_freeform_answers(
            moved_batch, max_new_tokens=max_new_tokens, num_beams=num_beams
        )
        layer1_texts = outputs.get("layer1_text")
        layer2_texts = outputs.get("layer2_text")
        if layer1_texts is None or layer2_texts is None:
            raise ValueError(
                "validation generation must return layer1_text and layer2_text"
            )
        segment_ids = batch["segment_id"]
        expected = len(segment_ids)
        if len(layer1_texts) != expected or len(layer2_texts) != expected:
            raise ValueError("validation generation batch length mismatch")
        for index, segment_id in enumerate(segment_ids):
            layer1_options = batch["layer1_options"][index]
            layer2_options = batch["layer2_options"][index]
            layer1_ref = batch["layer1_correct_option"][index]
            layer2_ref = batch["layer2_correct_option"][index]
            layer1_pred, _ = extract_mcq_option(layer1_texts[index], layer1_options)
            layer2_pred, _ = extract_mcq_option(layer2_texts[index], layer2_options)
            meter.update(
                category_id=batch["layer2_category_id"][index],
                category_name=batch["layer2_category_name"][index],
                segment_id=segment_id,
                stage1_correct=layer1_pred == layer1_ref,
                stage2_correct=layer2_pred == layer2_ref,
                stage1_pred=layer1_pred,
                stage1_ref=layer1_ref,
                stage2_pred=layer2_pred,
                stage2_ref=layer2_ref,
            )
    metrics = meter.compute()
    counts = meter.counts()
    metrics["overall"]["num_records"] = counts["overall"]["num_records"]
    for category_key, category_counts in counts["categories"].items():
        metrics["categories"][category_key]["num_records"] = category_counts[
            "num_records"
        ]
    if dist.is_available() and dist.is_initialized():
        counts = _category_metrics_to_count_tensor(metrics, device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        metrics = _category_count_tensor_to_metrics(counts)
    return metrics


def _checkpoint_payload(model, optimizer, epoch, step, extra_state=None):
    model = _unwrap_model(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    trainable_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }
    payload = {
        "model": trainable_state,
        "trainable_parameter_names": sorted(trainable_names),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if extra_state:
        payload.update(extra_state)
    return payload


def main(argv=None):
    args = parse_args(argv)
    distributed = init_distributed()
    seed_everything(args.seed + (dist.get_rank() if distributed else 0))
    if distributed and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    train_dataset = MiDashengSpatialQAPairDataset(
        args.train_json,
        args.feature_root,
        args.checkpoint_id,
        args.frontend_version,
        args.max_records,
    )
    if len(train_dataset) == 0:
        raise ValueError("empty training dataset")
    val_loader = None
    if args.val_json:
        val_dataset = MiDashengSpatialQAPairDataset(
            args.val_json,
            args.val_feature_root,
            args.checkpoint_id,
            args.frontend_version,
            args.val_max_records,
        )
        if len(val_dataset) == 0:
            raise ValueError("empty validation dataset")
        if distributed:
            val_loader = _make_loader(
                val_dataset, args, shuffle=False, distributed=True, train=False
            )
        else:
            val_loader = _make_loader(val_dataset, args, shuffle=False)

    base_model = SpatialMiDashengLMTrainingModel(
        args.midashenglm_root,
        enable_layer2=not args.disable_layer2,
        enable_llm_lora=not args.disable_llm_lora,
        llm_lora_r=args.llm_lora_r,
        llm_lora_alpha=args.llm_lora_alpha,
        llm_lora_dropout=args.llm_lora_dropout,
        llm_lora_num_layers=args.llm_lora_num_layers,
    ).to(device)
    if distributed:
        if device.type == "cuda":
            model = DDP(base_model, device_ids=[device.index], output_device=device.index)
        else:
            model = DDP(base_model)
    else:
        model = base_model
    current_trainable_names = {
        name
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    if distributed:
        train_loader = _make_loader(
            train_dataset, args, shuffle=True, distributed=True, train=True
        )
    else:
        train_loader = _make_loader(train_dataset, args, shuffle=True)

    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        init_state = init_checkpoint.get("model", init_checkpoint)
        missing, unexpected = base_model.load_state_dict(init_state, strict=False)
        if is_main_process():
            print(
                f"loaded_init_checkpoint={args.init_checkpoint} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    resumed_checkpoint = None
    if args.resume:
        resumed_checkpoint = torch.load(args.resume, map_location=device)
        saved_names = resumed_checkpoint.get("trainable_parameter_names")
        if saved_names is not None and set(saved_names) != current_trainable_names:
            raise ValueError(
                "resume checkpoint trainable_parameter_names do not match current model"
            )
        base_model.load_state_dict(
            resumed_checkpoint.get("model", resumed_checkpoint), strict=False
        )
        optimizer_state = resumed_checkpoint.get("optimizer")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(resumed_checkpoint.get("epoch", 0))
        global_step = int(resumed_checkpoint.get("step", 0))
        saved_best = resumed_checkpoint.get("best_metric_value")
        if saved_best is not None:
            best_metric = float(saved_best)
    if args.dry_run or start_epoch >= args.epochs:
        if distributed:
            cleanup_distributed()
        return

    select_best = val_loader is not None and not args.disable_layer2
    if (
        select_best
        and resumed_checkpoint is not None
        and "best_metric_value" in resumed_checkpoint
        and not (output_dir / "checkpoint.pt").exists()
        and is_main_process()
    ):
        torch.save(resumed_checkpoint, output_dir / "checkpoint.pt")
    validation_history = []
    last_payload = None
    for epoch in range(start_epoch, args.epochs):
        enable_layer2 = (
            not args.disable_layer2 and epoch + 1 > args.stage1_epochs
        )
        base_model.enable_layer2 = enable_layer2
        if is_main_process():
            print(f"epoch={epoch + 1} phase=train_start", flush=True)
        train_sampler = getattr(train_loader, "sampler", None)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(
            train_loader,
            disable=not is_main_process(),
            desc=(
                f"epoch {epoch + 1}/{args.epochs} "
                f"{'[layer1]' if not enable_layer2 else '[layer1+2]'}"
            ),
        )
        epoch_stats = {"loss": 0.0}
        for step, batch in enumerate(pbar, start=1):
            loss = model(_move(batch, device))["loss"]
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss")
            loss_value = float(loss.detach().cpu())
            epoch_stats["loss"] += loss_value
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            pbar.set_postfix(loss=f"{loss_value:.4f}")
        if len(train_loader) % args.grad_accum:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        if is_main_process():
            dump_json(
                {"loss": epoch_stats["loss"] / max(len(train_loader), 1)},
                output_dir / f"train_epoch_{epoch + 1:03d}.json",
            )
            print(
                f"epoch={epoch + 1} phase=train_complete global_step={global_step}",
                flush=True,
            )

        val_metrics = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    val_loss = model(_move(batch, device))["loss"]
                    if not torch.isfinite(val_loss):
                        raise ValueError("non-finite validation loss")
                    val_losses.append(float(val_loss.detach().cpu()))
            if enable_layer2:
                val_metrics = _evaluate_stage2_accuracy(
                    model,
                    val_loader,
                    device,
                    max_new_tokens=args.val_max_new_tokens,
                    num_beams=args.val_num_beams,
                )
            val_loss_total = torch.tensor(sum(val_losses), dtype=torch.float64, device=device)
            val_count = torch.tensor(len(val_losses), dtype=torch.long, device=device)
            if distributed:
                dist.all_reduce(val_loss_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
            val_mean = float(val_loss_total.cpu()) / max(int(val_count.cpu()), 1)
            val_message = f"epoch={epoch + 1} val_loss={val_mean:.6f}"
            if val_metrics is not None:
                val_message += f" grounded_stage2_acc={val_metrics['overall']['grounded_stage2_acc']:.6f}"
            if is_main_process():
                print(val_message, flush=True)
                if val_metrics is not None:
                    validation_history.append(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "selection_metric_name": "overall.grounded_stage2_acc",
                            "selection_metric_value": float(
                                val_metrics["overall"]["grounded_stage2_acc"]
                            ),
                            "metrics": val_metrics,
                        }
                    )
                    dump_json(
                        val_metrics, output_dir / f"val_epoch_{epoch + 1:03d}.json"
                    )
                    dump_json(
                        validation_history, output_dir / "validation_history.json"
                    )
            model.train()

        epoch_extra = {}
        if val_metrics is not None:
            epoch_extra = {
                "val_metrics": val_metrics,
                "selection_metric_name": "overall.grounded_stage2_acc",
                "selection_metric_value": float(val_metrics["overall"]["grounded_stage2_acc"]),
            }
        epoch_payload = _checkpoint_payload(
            model, optimizer, epoch + 1, global_step, epoch_extra
        )
        if is_main_process():
            torch.save(epoch_payload, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")
        last_payload = epoch_payload

        if select_best:
            score = float(val_metrics["overall"]["grounded_stage2_acc"])
            if score > best_metric:
                best_metric = score
                best_payload = dict(epoch_payload)
                best_payload.update(
                    {
                        "best_metric_name": "overall.grounded_stage2_acc",
                        "best_metric_value": score,
                    }
                )
                if is_main_process():
                    torch.save(best_payload, output_dir / "checkpoint.pt")

        if distributed:
            dist.barrier()

    if not select_best and last_payload is not None and is_main_process():
        torch.save(last_payload, output_dir / "checkpoint.pt")
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
