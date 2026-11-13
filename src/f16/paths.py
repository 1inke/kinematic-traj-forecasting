from __future__ import annotations
import os
from pathlib import Path

DEFAULTS={
 'cache':'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',
 'labels':'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',
 'metadata':'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz',
 'splits':'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz',
 'encoder':'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',
 'outputs':'experiments'}

def project_root():return Path(os.environ.get('F16_ROOT',Path(__file__).resolve().parents[2])).resolve()

def resolve_paths(config=None):
    root=project_root();values=DEFAULTS.copy();path=Path(config) if config else root/'config/paths.yaml'
    if path.exists():
        # Deliberately small scalar-only YAML subset; no implicit object loader.
        for lineno,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
            line=line.split('#',1)[0].strip()
            if not line:continue
            if ':' not in line:raise ValueError(f'Invalid scalar path YAML line {lineno}')
            key,value=line.split(':',1);key=key.strip();value=value.strip().strip('\"\'')
            if key not in set(DEFAULTS)|{'root','astra_root'}:raise ValueError(f'Unknown path key {key}')
            values[key]=os.path.expandvars(value)
        if 'root' in values and 'F16_ROOT' not in os.environ:
            candidate=Path(values.pop('root'));root=(path.parent.parent/candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    result={key:(Path(v) if Path(v).is_absolute() else root/v).resolve() for key,v in values.items() if key!='root'}
    if 'F16_ASTRA_ROOT' in os.environ:result['astra_root']=Path(os.environ['F16_ASTRA_ROOT']).resolve()
    result['root']=root
    return result
