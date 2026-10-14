from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .layered_dataset import SpatialQAPairDataset
from .midashenglm_cache import load_midasheng_cache

class MiDashengSpatialQAPairDataset(SpatialQAPairDataset):
    def __init__(self, json_path, feature_root, expected_checkpoint_id=None, expected_frontend_version="dasheng-64mel-v1", max_records=None, task_id=None):
        if max_records is not None and max_records <= 0: raise ValueError("max_records must be positive when provided")
        self.json_path=Path(json_path); self.feature_root=Path(feature_root); self.task_id=task_id
        self.expected_checkpoint_id=expected_checkpoint_id; self.expected_frontend_version=expected_frontend_version
        self.records=self._load_records(max_records=max_records)
    def __getitem__(self, index):
        r=self.records[index]; p=self._feature_path(r.segment_id)
        if not p.is_file(): raise FileNotFoundError(p)
        item=load_midasheng_cache(p, self.expected_checkpoint_id, self.expected_frontend_version)
        return {**item, "segment_id":r.segment_id,"audio_path":r.audio_path,"layer1_question":r.layer1_question,"layer1_answer":r.layer1_answer,"layer2_question":r.layer2_question,"layer2_answer":r.layer2_answer,"layer1_options":r.layer1_options,"layer1_correct_option":r.layer1_correct_option,"layer2_options":r.layer2_options,"layer2_correct_option":r.layer2_correct_option,"layer2_category_id":r.layer2_category_id,"layer2_category_name":r.layer2_category_name,"task_id":r.task_id}

def collate_midasheng_spatial(batch):
    if not batch: raise ValueError("batch must be non-empty")
    native=[x.get("midasheng_audio_tokens") for x in batch]; has=[x is not None for x in native]
    if any(has) and not all(has): raise ValueError("mixed native token presence")
    b=len(batch); max_t=max(x["dasheng_features"].shape[0] for x in batch); d=batch[0]["dasheng_features"].shape[1]
    raw=torch.zeros(b,max_t,d,dtype=torch.float32); mask=torch.zeros(b,max_t,dtype=torch.long)
    for i,x in enumerate(batch):
        t=x["dasheng_features"].shape[0]; raw[i,:t,:]=torch.as_tensor(x["dasheng_features"]); mask[i,:t]=torch.as_tensor(x["dasheng_attention_mask"])
    out={"dasheng_features":raw,"dasheng_attention_mask":mask,"spatial_features":torch.stack([torch.as_tensor(x["spatial_features"]) for x in batch]),"metadata":[x["metadata"] for x in batch]}
    if all(has):
        mt=max(x.shape[0] for x in native); nt=torch.zeros(b,mt,native[0].shape[1]); nm=torch.zeros(b,mt,dtype=torch.long)
        for i,x in enumerate(native): nt[i,:x.shape[0]]=torch.as_tensor(x); nm[i,:x.shape[0]]=1
        out.update(midasheng_audio_tokens=nt, midasheng_audio_attention_mask=nm)
    else: out["midasheng_audio_tokens"]=None
    for k in ("segment_id","audio_path","layer1_question","layer1_answer","layer2_question","layer2_answer","layer1_options","layer1_correct_option","layer2_options","layer2_correct_option","layer2_category_id","layer2_category_name","task_id"): out[k]=[x.get(k) for x in batch]
    return out
