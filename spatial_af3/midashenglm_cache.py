from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import os, tempfile
from pathlib import Path
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class MiDashengCacheMetadata:
    checkpoint_id: str
    sample_rate: int = 16000
    num_mel_bins: int = 64
    subsample_factor: int = 5
    raw_dim: int = 1280
    native_dim: int = 3584
    spatial_dim: int = 448
    raw_frames: int = 0
    native_frames: int = 0
    dtype: str = "float32"
    frontend_version: str = "dasheng-64mel-v1"
    schema_version: int = 1

    def __post_init__(self):
        if not self.checkpoint_id.strip(): raise ValueError("checkpoint_id must be non-empty")
        if not self.frontend_version.strip(): raise ValueError("frontend_version must be non-empty")
        if self.schema_version != 1: raise ValueError("schema_version must be 1")
        for n in ("sample_rate","num_mel_bins","subsample_factor","raw_dim","native_dim","spatial_dim"):
            if getattr(self,n) <= 0: raise ValueError(f"{n} must be positive")
        if self.raw_frames < 1: raise ValueError("raw_frames must be >= 1")
        if self.native_frames < 0: raise ValueError("native_frames must be >= 0")

def _validate(meta, raw, mask, spatial, native):
    try: expected_dtype = np.dtype(meta.dtype)
    except TypeError as exc: raise ValueError("metadata dtype is invalid") from exc
    for arr, name in ((raw,"dasheng_features"),(spatial,"spatial_features")):
        if not np.issubdtype(arr.dtype, np.floating) or not np.isfinite(arr).all(): raise ValueError(f"{name} must be finite floating dtype")
    if native is not None and (not np.issubdtype(native.dtype, np.floating) or not np.isfinite(native).all()): raise ValueError("midasheng_audio_tokens must be finite floating dtype")
    if raw.dtype != expected_dtype or spatial.dtype != expected_dtype or (native is not None and native.dtype != expected_dtype): raise ValueError("feature dtype does not match metadata dtype")
    if raw.ndim != 2 or raw.shape[1] != meta.raw_dim: raise ValueError("dasheng_features must have shape [T, raw_dim]")
    if mask.ndim != 1 or mask.shape[0] != raw.shape[0] or (not np.issubdtype(mask.dtype, np.integer) and mask.dtype != np.bool_) or not np.all((mask == 0) | (mask == 1)): raise ValueError("dasheng_attention_mask shape/dtype/value mismatch")
    if spatial.ndim != 2 or spatial.shape != (500, meta.spatial_dim): raise ValueError("spatial_features must have shape [500, spatial_dim]")
    if native is not None and (native.ndim != 2 or native.shape[1] != meta.native_dim): raise ValueError("midasheng_audio_tokens shape mismatch")
    if meta.raw_frames != raw.shape[0] or (native is None and meta.native_frames != 0) or (native is not None and meta.native_frames != native.shape[0]): raise ValueError("metadata frame count mismatch")

def save_midasheng_cache(path, dasheng_features, dasheng_attention_mask, spatial_features, metadata: MiDashengCacheMetadata, midasheng_audio_tokens=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(metadata, MiDashengCacheMetadata): raise TypeError("metadata must be MiDashengCacheMetadata")
    raw=np.asarray(dasheng_features); mask=np.asarray(dasheng_attention_mask); spatial=np.asarray(spatial_features); native=None if midasheng_audio_tokens is None else np.asarray(midasheng_audio_tokens)
    _validate(metadata, raw, mask, spatial, native)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent)); os.close(fd)
    try:
        np.savez_compressed(tmp, dasheng_features=raw, dasheng_attention_mask=mask, spatial_features=spatial, **({} if native is None else {"midasheng_audio_tokens": native}), metadata=np.array(json.dumps(asdict(metadata))))
        os.replace(tmp + ".npz", path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
        if os.path.exists(tmp + ".npz"): os.unlink(tmp + ".npz")

def load_midasheng_cache(path, expected_checkpoint_id: Optional[str] = None, expected_frontend_version: Optional[str] = None):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required={"dasheng_features","dasheng_attention_mask","spatial_features","metadata"}
        missing=required-set(data.files)
        if missing: raise ValueError(f"cache missing fields: {sorted(missing)}")
        try: meta=MiDashengCacheMetadata(**json.loads(str(data["metadata"].item())))
        except Exception as exc: raise ValueError("invalid cache metadata") from exc
        if expected_checkpoint_id is not None and meta.checkpoint_id != expected_checkpoint_id: raise ValueError("checkpoint mismatch")
        if expected_frontend_version is not None and meta.frontend_version != expected_frontend_version: raise ValueError("frontend version mismatch")
        native=data["midasheng_audio_tokens"] if "midasheng_audio_tokens" in data.files else None
        if native is not None and native.size == 0: native = None
        _validate(meta, data["dasheng_features"], data["dasheng_attention_mask"], data["spatial_features"], native)
        return {"dasheng_features": data["dasheng_features"], "dasheng_attention_mask": data["dasheng_attention_mask"], "spatial_features": data["spatial_features"], "midasheng_audio_tokens": native, "metadata": meta}
