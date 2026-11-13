"""New execution only. No imports of historical experiment entry points."""
from pathlib import Path
import sys, json, hashlib
from datetime import datetime, timezone, timedelta
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from src.f16.io import sha256,write_json,read_json
from src.f16.resource import Guard
SEEDS=(42,43,44)
NAMES=('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')
ESTIMATE=3*1024**3
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def verify():
    m=read_json(HERE/'source_manifest.json')
    for k,v in m['files'].items():
        if sha256(ROOT/k)!=v['sha256']:raise ValueError('SOURCE_SHA '+k)
    if digest({k:v['sha256'] for k,v in m['files'].items()})!=m['digest']:raise ValueError('SOURCE_DIGEST')
    return m['digest']
def stage(name,**kw):write_json(HERE/'runtime_state.json',dict(status='RUNNING',stage=name,at=now(),**kw))
def validate_done(path,source):
    d=read_json(path)
    if d['source_digest']!=source:raise ValueError('Completed source mismatch')
    for relative,expected in d['outputs'].items():
        if sha256(path.parent/relative)!=expected:raise ValueError('Completed output SHA mismatch '+relative)
    return d
