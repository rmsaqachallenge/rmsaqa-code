#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

from spatial_af3.dataset import SpatialQAPairDataset, qa_pair_collate_fn
from spatial_af3.evaluation import evaluate_spatial_model
from spatial_af3.models import SpatialAF3GenerativeModel
from spatial_af3.utils import dump_json, ensure_dir, load_trainable_state_dict

DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_JSON = DATA_ROOT / "dev_QA"


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
    parser = argparse.ArgumentParser(
        description="Evaluate frozen-AF3 spatial sidecar with AF3-style MCQ option extraction."
    )
    parser.add_argument("--af3-root", required=True)
    parser.add_argument(
        "--test-json",
        default=str(DEFAULT_TEST_JSON),
        help="Evaluation split as a legacy aggregate JSON file or a per-sample JSON directory.",
    )
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--extract-features-on-the-fly", action="store_true")
    parser.add_argument("--write-missing-features", action="store_true")
    parser.add_argument("--audio-token-stride", type=int, default=3, help="Spatial feature resampling stride used during preprocessing/loading. Keep 3 so spatial tokens stay at 500 and match the truncated AF-Whisper branch.")
    parser.add_argument("--disable-stage35", action="store_true")
    parser.add_argument("--disable-llm-lora", action="store_true", help="Disable trainable LLM LoRA adapters.")
    parser.add_argument("--llm-lora-r", type=int, default=16, help="LoRA rank for AF3/Qwen2 LLM adapters.")
    parser.add_argument("--llm-lora-alpha", type=int, default=16, help="LoRA alpha for AF3/Qwen2 LLM adapters.")
    parser.add_argument("--llm-lora-dropout", type=float, default=0.05, help="LoRA dropout for AF3/Qwen2 LLM adapters.")
    parser.add_argument("--llm-lora-num-layers", type=int, default=28, help="Apply LLM LoRA to the first N transformer layers. Default 28 = all layers.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--answers-log-name", default="answers_log.jsonl", help="Per-sample answer log filename under output-dir.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--audio-encoder-device",
        default="cpu",
        help="Device for frozen AF3 sound_tower during evaluation. Default cpu to reduce GPU memory pressure.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.test_json = str(ensure_foa_json(args.test_json, "--test-json"))
    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)
    dump_json(vars(args), output_dir / "eval_args.json")

    dataset = SpatialQAPairDataset(
        json_path=args.test_json,
        feature_root=args.feature_root,
        extract_features_on_the_fly=args.extract_features_on_the_fly,
        write_missing_features=args.write_missing_features,
        audio_token_stride=args.audio_token_stride,
        max_records=args.max_records,
    )
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=qa_pair_collate_fn,
        pin_memory=True,
    )

    model = SpatialAF3GenerativeModel(
        af3_root=args.af3_root,
        apply_stage35=not args.disable_stage35,
        enable_llm_lora=not args.disable_llm_lora,
        llm_lora_r=args.llm_lora_r,
        llm_lora_alpha=args.llm_lora_alpha,
        llm_lora_dropout=args.llm_lora_dropout,
        llm_lora_num_layers=args.llm_lora_num_layers,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    missing_nonrequired_keys, unexpected_keys = load_trainable_state_dict(
        model,
        ckpt["state_dict"],
        checkpoint_format_version=ckpt.get("checkpoint_format_version"),
    )
    print(
        f"missing_nonrequired_keys={len(missing_nonrequired_keys)} "
        f"unexpected_keys={len(unexpected_keys)}"
    )
    if args.audio_encoder_device:
        model.backbone.audio_encoder.to(torch.device(args.audio_encoder_device))
    log_path = output_dir / args.answers_log_name
    all_metrics, pred_records = evaluate_spatial_model(
        model,
        loader,
        device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        desc="eval",
        answers_log_path=log_path,
    )
    dump_json(all_metrics, output_dir / "metrics.json")
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(pred_records, f, ensure_ascii=False, indent=2)
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
