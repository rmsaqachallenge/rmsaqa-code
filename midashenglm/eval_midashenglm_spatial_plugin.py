#!/usr/bin/env python
"""Cache-first MiDashengLM spatial evaluation entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Keep argument parsing usable in lightweight environments.
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from spatial_af3.midashenglm_eval_dataset import MiDashengEvalDataset, collate_midashenglm_eval
    from spatial_af3.midashenglm_eval_models import SpatialMiDashengLMEvaluator
    from spatial_af3.midashenglm_evaluation import evaluate_midashenglm_model
except ModuleNotFoundError:  # pragma: no cover - exercised only without ML dependencies
    torch = None
    DataLoader = SequentialSampler = None
    MiDashengEvalDataset = collate_midashenglm_eval = None
    SpatialMiDashengLMEvaluator = None
    evaluate_midashenglm_model = None


def parse_args(argv=None):
    positive = lambda value: (v if (v := int(value)) > 0 else (_ for _ in ()).throw(argparse.ArgumentTypeError("must be positive")))
    p = argparse.ArgumentParser(description="Evaluate the trained MiDashengLM spatial model.")
    p.add_argument("--midashenglm-root", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--feature-root", default="")
    p.add_argument("--audio-root", default="")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=positive, default=4)
    p.add_argument("--num-workers", type=lambda value: (v if (v := int(value)) >= 0 else (_ for _ in ()).throw(argparse.ArgumentTypeError("must be non-negative"))), default=0)
    p.add_argument("--max-records", type=positive, default=None)
    p.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    p.add_argument("--audio-encoder-device", default="cpu")
    p.add_argument("--max-new-tokens", type=positive, default=12)
    p.add_argument("--num-beams", type=positive, default=1)
    p.add_argument("--checkpoint-id", default=None)
    p.add_argument("--frontend-version", default="dasheng-64mel-v1")
    p.add_argument("--online-waveform", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--answers-log-name", default="answers_log.jsonl")
    args = p.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    if torch is None:
        raise RuntimeError("MiDashengLM evaluation requires torch and project dependencies")
    out = Path(args.output_dir)
    for label, value in (("midashenglm-root", args.midashenglm_root), ("test-json", args.test_json)):
        if not Path(value).exists():
            raise FileNotFoundError(f"--{label} does not exist: {value}")
    if not args.online_waveform and (not args.feature_root or not Path(args.feature_root).exists()):
        raise FileNotFoundError(f"--feature-root does not exist: {args.feature_root}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"--checkpoint does not exist: {args.checkpoint}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    dataset = MiDashengEvalDataset(
        json_path=args.test_json, feature_root=args.feature_root,
        expected_checkpoint_id=args.checkpoint_id,
        expected_frontend_version=args.frontend_version,
        max_records=args.max_records,
        online_waveform=args.online_waveform,
        extract_spatial=True,
        audio_root=args.audio_root,
    )
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate_midashenglm_eval)
    kwargs = {"online_waveform": args.online_waveform, "audio_encoder_device": args.audio_encoder_device}
    adapter = SpatialMiDashengLMEvaluator(args.midashenglm_root, checkpoint=args.checkpoint, **kwargs)
    metrics, _ = evaluate_midashenglm_model(
        adapter, loader, torch.device(args.device), args.max_new_tokens, args.num_beams,
        output_dir=out, answers_log_path=out / args.answers_log_name)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
