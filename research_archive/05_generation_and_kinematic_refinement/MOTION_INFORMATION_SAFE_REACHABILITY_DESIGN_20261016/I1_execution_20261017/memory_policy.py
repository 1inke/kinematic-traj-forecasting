"""Versioned task1.3 guard: current AND OS-retained lifetime peaks. No OS changes."""
import ctypes,json,os,math,hashlib
from pathlib import Path
from datetime import datetime
from collections import deque
import psutil
H=Path(__file__).resolve().parent
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
RESOURCE=dict(policy='task_memory_1p3_peak_v2',peak_private_bytes=0,peak_RSS_bytes=0,min_available_physical_bytes=None,min_available_commit_bytes=None,checks=0)
_config=None;_stage='IMPORT';_observations=deque(maxlen=128);_last_written_peak=0
def write(name,value):
    p=H/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def set_stage(name):
    global _stage
    _stage=name
def config():
    global _config
    if _config is None:
        p=H/'resource_policy.json';raw=p.read_bytes();a=json.loads(raw);ep=H/a['estimate_file'];eraw=ep.read_bytes();e=json.loads(eraw)
        assert a['safety_factor']==1.3 and a['peak_limit_enforced'] is True
        _config=(int(e['estimated_total_peak_bytes']),hashlib.sha256(raw).hexdigest(),hashlib.sha256(eraw).hexdigest())
    return _config
def snapshot():
    x=Mem();x.length=ctypes.sizeof(x);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(x));p=psutil.Process().memory_info()
    return dict(observed_at=datetime.now().astimezone().isoformat(),stage_at_observation=_stage,physical_available_bytes=int(x.avail_phys),commit_available_bytes=int(x.avail_page),private_bytes=int(p.private),RSS_bytes=int(p.rss),peak_private_bytes=int(p.peak_pagefile),peak_RSS_bytes=int(p.peak_wset))
def decide(s,e,factor=1.3,mode='runtime'):
    assert factor==1.3 and mode in ('startup','runtime') and math.isfinite(e) and e>0
    for k in ('physical_available_bytes','commit_available_bytes','private_bytes','RSS_bytes','peak_private_bytes','peak_RSS_bytes'):assert math.isfinite(s[k]) and s[k]>=0
    pn=e if mode=='startup' else max(e-s['RSS_bytes'],0);cn=e if mode=='startup' else max(e-s['private_bytes'],0);limit=factor*e;reasons=[]
    if s['physical_available_bytes']<max(6*2**30,factor*pn):reasons.append('AVAILABLE_PHYSICAL_LT_1P3_ESTIMATED_REMAINING')
    if s['commit_available_bytes']<max(6*2**30,factor*cn):reasons.append('AVAILABLE_COMMIT_LT_1P3_ESTIMATED_REMAINING')
    if max(s['private_bytes'],s['RSS_bytes'],s['peak_private_bytes'],s['peak_RSS_bytes'])>limit:reasons.append('CURRENT_OR_RETAINED_PEAK_EXCEEDS_1P3_E')
    return dict(status='PASS' if not reasons else 'STOP_RESOURCE',estimated_total_peak_bytes=e,factor=factor,mode=mode,required_available_physical_bytes=factor*pn,required_available_commit_bytes=factor*cn,dynamic_process_limit_bytes=limit,reasons=reasons,snapshot=s,fixed6GiB_rule=False)
def guard(mode='runtime'):
    global _last_written_peak
    e,ph,eh=config();s=snapshot();r=decide(s,e,mode=mode);RESOURCE['checks']+=1
    for k in ('peak_private_bytes','peak_RSS_bytes'):RESOURCE[k]=max(RESOURCE[k],s[k])
    for k,sk in (('min_available_physical_bytes','physical_available_bytes'),('min_available_commit_bytes','commit_available_bytes')):RESOURCE[k]=s[sk] if RESOURCE[k] is None else min(RESOURCE[k],s[sk])
    RESOURCE.update(estimated_total_peak_bytes=e,limit_bytes=1.3*e,policy_SHA=ph,estimate_SHA=eh)
    peak=max(RESOURCE['peak_private_bytes'],RESOURCE['peak_RSS_bytes']);changed=not _observations or _observations[-1]['stage_at_observation']!=_stage
    if changed or mode=='startup' or peak-_last_written_peak>=16*2**20 or r['status']!='PASS':
        _observations.append(s);write('RESOURCE_OBSERVATIONS.json',dict(observations=list(_observations),bounded_last=128,peak_occurrence_time_unknown=True,resource=RESOURCE));_last_written_peak=peak
    if r['status']!='PASS':
        write('RESOURCE_FAILURE.json',dict(**r,resource=dict(RESOURCE)));raise MemoryError('STOP_RESOURCE '+','.join(r['reasons']))
    return r
