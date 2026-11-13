"""Strict split/identity contracts. Groups and labels are NOT inference features."""
from dataclasses import dataclass
import numpy as np
from .paths import resolve_paths
from .io import read_json,verify_manifest

SIZES={'FIT':18909,'inner':1856,'outer':1035,'original_CAL':2879}

def split_checks(ids,track,splits):
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate sample IDs')
    pairs={};names=list(splits)
    for name,ix in splits.items():
        if len(ix)!=SIZES[name] or len(np.unique(ix))!=len(ix) or np.any(ix<0) or np.any(ix>=len(ids)):raise ValueError(f'Invalid split {name}')
    for i,a in enumerate(names):
        for b in names[i+1:]:
            common=np.intersect1d(ids[splits[a]],ids[splits[b]])
            if len(common):raise ValueError(f'Cross-split sample IDs: {a}/{b}')
            shared=np.intersect1d(track[splits[a]],track[splits[b]])
            pairs[a+'__'+b]={'shared_tracks':len(shared),'track_keys':shared.tolist()}
    return pairs

def load_identities(paths=None,verify_frozen=True):
    paths=paths or resolve_paths();root=paths['root']
    if verify_frozen:
        manifest=read_json(root/'config/data_manifest.v1.json');verify_manifest(manifest['files'],root)
    with np.load(paths['cache'],allow_pickle=False) as z:ids=z['sample_id'].copy();groups=z['change_group'].astype(int)
    with np.load(paths['metadata'],allow_pickle=False) as z:
        sorted_order=np.argsort(z['sample_id']);found=np.searchsorted(z['sample_id'][sorted_order],ids)
        if np.any(found>=len(sorted_order)):raise ValueError('Missing metadata ID')
        loc=sorted_order[found]
        if not np.array_equal(z['sample_id'][loc],ids):raise ValueError('Metadata identity mismatch')
        md={k:z[k][loc].copy() for k in z.files}
    with np.load(paths['splits'],allow_pickle=False) as z:
        if not np.array_equal(z['sample_id'],ids[:25912]):raise ValueError('Registry mismatch')
        split={name:z[key].copy() for name,key in [('FIT','grouped_train_index'),('inner','inner_cal_index'),('outer','outer_eval_index')]}
        for name,key in [('FIT','grouped_train_ids'),('inner','inner_cal_ids'),('outer','outer_eval_ids')]:
            if not np.array_equal(ids[split[name]],z[key]):raise ValueError(f'Split ID/index mismatch {name}')
    split['original_CAL']=np.arange(25912,28791)
    track=np.array([f'{s}::{int(p)}' for s,p in zip(md['sequence'],md['pedestrian_id'])]);pair=split_checks(ids,track,split)
    return dict(sample_id=ids,groups=groups,track=track,metadata=md,split=split,pair_overlap=pair)

@dataclass(frozen=True)
class ModelInput:
    observed: np.ndarray
    candidates: np.ndarray | None = None

def inference_input(observed,candidates=None):
    # Separate explicit API, no **kwargs accepting GT/groups/IDs unnoticed.
    obs=np.asarray(observed)
    if not np.isfinite(obs).all():raise ValueError('Nonfinite observation')
    if candidates is not None and not np.isfinite(candidates).all():raise ValueError('Nonfinite candidates')
    return ModelInput(obs.copy(),None if candidates is None else np.asarray(candidates).copy())
