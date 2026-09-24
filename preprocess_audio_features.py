#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from spatial_af3.feature_ops import AudioFeatureConfig, AudioFeatureExtractor, save_feature_npz
from spatial_af3.utils import ensure_dir

_THREAD_LOCAL = threading.local()


def _get_thread_extractor(audio_token_stride: int) -> AudioFeatureExtractor:
    extractor = getattr(_THREAD_LOCAL, "extractor", None)
    cached_stride = getattr(_THREAD_LOCAL, "audio_token_stride", None)
    if extractor is None or cached_stride != audio_token_stride:
        extractor = AudioFeatureExtractor(AudioFeatureConfig(audio_token_stride=audio_token_stride))
        _THREAD_LOCAL.extractor = extractor
        _THREAD_LOCAL.audio_token_stride = audio_token_stride
    return extractor


def _extract_one(
    item: dict,
    output_dir: str,
    audio_token_stride: int,
    overwrite: bool,
    compressed: bool,
) -> str:
    output_dir_path = Path(output_dir)
    segment_id = item["segment_id"]
    out_path = output_dir_path / f"{segment_id}.npz"
    if out_path.exists() and not overwrite:
        return segment_id
    extractor = _get_thread_extractor(audio_token_stride)
    data = extractor.extract_from_path(item["audio_path"])
    save_feature_npz(data, out_path, compressed=compressed)
    return segment_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute FOA spatial features and 10 s / 1000-frame mono Whisper input features.")
    parser.add_argument("--json", required=True, help="Path to train/test json.")
    parser.add_argument("--output-dir", required=True, help="Directory to save per-segment .npz feature files.")
    parser.add_argument("--audio-token-stride", type=int, default=3, help="Spatial feature resampling stride. Keep 3 so the spatial branch stays at 500 frames and matches the truncated AF-Whisper branch.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="Process workers for parallel preprocessing. 0 means single-process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    records = payload["records"]
    if args.max_records is not None:
        records = records[: args.max_records]

    output_dir = ensure_dir(args.output_dir)
    if args.num_workers <= 0:
        extractor = AudioFeatureExtractor(AudioFeatureConfig(audio_token_stride=args.audio_token_stride))
        pbar = tqdm(records, desc="extract-features")
        for item in pbar:
            segment_id = item["segment_id"]
            out_path = output_dir / f"{segment_id}.npz"
            if out_path.exists() and not args.overwrite:
                continue
            data = extractor.extract_from_path(item["audio_path"])
            save_feature_npz(data, out_path, compressed=args.compressed)
            pbar.set_postfix(segment_id=segment_id)
        return

    pbar = tqdm(total=len(records), desc="extract-features")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        mapped = executor.map(
            _extract_one,
            records,
            [str(output_dir)] * len(records),
            [args.audio_token_stride] * len(records),
            [args.overwrite] * len(records),
            [args.compressed] * len(records),
            chunksize=max(1, min(64, len(records) // max(1, args.num_workers * 8))),
        )
        for segment_id in mapped:
            pbar.update(1)
            pbar.set_postfix(segment_id=segment_id)
    pbar.close()


if __name__ == "__main__":
    main()
