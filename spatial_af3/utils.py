from __future__ import annotations

import json
import os
import random
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


TRAINABLE_CHECKPOINT_FORMAT_VERSION = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def trainable_state_names(model: torch.nn.Module) -> set[str]:
    names = trainable_parameter_names(model)
    persistent_state_names = set(model.state_dict())
    for module_name, module in model.named_modules():
        if not any(
            parameter.requires_grad
            for parameter in module.parameters(recurse=False)
        ):
            continue
        for buffer_name, _ in module.named_buffers(recurse=False):
            qualified_name = (
                f"{module_name}.{buffer_name}" if module_name else buffer_name
            )
            if qualified_name in persistent_state_names:
                names.add(qualified_name)
    return names


def trainable_parameter_names(model: torch.nn.Module) -> set[str]:
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def load_trainable_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_format_version: int | None = None,
):
    version = 1 if checkpoint_format_version is None else checkpoint_format_version
    if version == TRAINABLE_CHECKPOINT_FORMAT_VERSION:
        required_names = trainable_state_names(model)
        format_label = f"Checkpoint format version {version}"
        state_kind = "trainable state"
    elif version == 1:
        required_names = trainable_parameter_names(model)
        format_label = "legacy checkpoint format version 1"
        state_kind = "trainable parameter"
    else:
        raise ValueError(f"Unsupported checkpoint format version: {version!r}.")

    missing = sorted(required_names.difference(state_dict))
    if missing:
        raise RuntimeError(
            f"{format_label} state_dict is missing {len(missing)} required "
            f"{state_kind} entries: {', '.join(missing)}"
        )

    if version == 1:
        buffer_names = trainable_state_names(model).difference(required_names)
        defaulted_buffers = sorted(buffer_names.difference(state_dict))
        if defaulted_buffers:
            warnings.warn(
                f"Legacy checkpoint format version 1 omits "
                f"{len(defaulted_buffers)} trainable module buffers; model default "
                f"buffer values will be used: {', '.join(defaulted_buffers)}",
                RuntimeWarning,
                stacklevel=2,
            )
    return model.load_state_dict(state_dict, strict=False)


def init_distributed() -> bool:
    if is_dist_initialized():
        return True
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank is None or world_size is None or local_rank is None:
        return False
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))
    return True


def cleanup_distributed() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()
