import argparse, json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from spatial_af3.layered_dataset import SpatialQAPairDataset
from spatial_af3.feature_ops import AudioFeatureExtractor, AudioFeatureConfig
from spatial_af3.models.midashenglm_generative_backbone import FrozenMiDashengLMGenerativeBackbone
from spatial_af3.midashenglm_cache import MiDashengCacheMetadata, save_midasheng_cache

def parse_args(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--json',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--input-root'); p.add_argument('--device',default='cpu'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--cache-native',action='store_true'); p.add_argument('--max-records',type=int); p.add_argument('--cache-dtype',choices=['float32','float16'],default='float32'); return p.parse_args(argv)
def main(argv=None):
 a=parse_args(argv); ds=SpatialQAPairDataset(a.json,max_records=a.max_records); ex=AudioFeatureExtractor(AudioFeatureConfig()); bb=FrozenMiDashengLMGenerativeBackbone(a.checkpoint,enable_llm_lora=False).to(a.device); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); input_root=Path(a.input_root) if a.input_root else Path(a.json).resolve().parent
 for r in ds.records:
  p=out/(r.segment_id+'.npz')
  if p.exists() and not a.overwrite: continue
  audio_path=Path(r.audio_path); audio_path=audio_path if audio_path.is_absolute() else input_root/audio_path
  wave=ex.load_audio(audio_path); spatial_fp32=ex.extract_spatial_features(wave).cpu().numpy()
  if not np.isfinite(spatial_fp32).all(): raise ValueError(f"spatial features are non-finite before {a.cache_dtype} cast for {r.segment_id}")
  spatial=spatial_fp32.astype(a.cache_dtype)
  if not np.isfinite(spatial).all(): raise ValueError(f"spatial features overflow during {a.cache_dtype} cast for {r.segment_id}; use --cache-dtype float32")
  mono=wave[0].unsqueeze(0); mono=torchaudio.functional.resample(mono, 24000, 16000); valid_samples=max(640,min(mono.shape[-1],160000)); mono=AudioFeatureExtractor._pad_or_trim(mono,160000)
  bundle=bb._encode_audio_bundle(mono.to(a.device),torch.tensor([valid_samples],device=a.device),include_native=a.cache_native); raw_fp32=bundle['raw_features'][0].cpu().numpy()
  if not np.isfinite(raw_fp32).all(): raise ValueError(f"dasheng features are non-finite before {a.cache_dtype} cast for {r.segment_id}")
  raw=raw_fp32.astype(a.cache_dtype)
  if not np.isfinite(raw).all(): raise ValueError(f"dasheng features overflow during {a.cache_dtype} cast for {r.segment_id}; use --cache-dtype float32")
  mask=bundle['raw_mask'][0].cpu().numpy() if bundle['raw_mask'] is not None else torch.ones(raw.shape[0],dtype=torch.int8).numpy(); native=None
  if bundle['native_tokens'] is not None:
   native_fp32=bundle['native_tokens'][0].cpu().numpy()
   if not np.isfinite(native_fp32).all(): raise ValueError(f"native tokens are non-finite before {a.cache_dtype} cast for {r.segment_id}")
   native=native_fp32.astype(a.cache_dtype)
   if not np.isfinite(native).all(): raise ValueError(f"native tokens overflow during {a.cache_dtype} cast for {r.segment_id}; use --cache-dtype float32")
  meta=MiDashengCacheMetadata(checkpoint_id=str(a.checkpoint),raw_frames=raw.shape[0],native_frames=0 if native is None else native.shape[0],dtype=a.cache_dtype)
  save_midasheng_cache(p,raw,mask,spatial,meta,native)
if __name__=='__main__': main()
