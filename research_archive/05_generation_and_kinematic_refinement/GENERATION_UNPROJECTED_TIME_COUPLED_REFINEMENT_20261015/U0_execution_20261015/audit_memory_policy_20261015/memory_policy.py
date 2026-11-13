"""Human-approved task-relative memory guard, no fixed6GiB rule."""
import ctypes,json,os,math
from pathlib import Path
from datetime import datetime
import psutil
H=Path(__file__).resolve().parent
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
RESOURCE=dict(peak_private_bytes=0,peak_RSS_bytes=0,min_available_physical_bytes=None,min_available_commit_bytes=None,checks=0)

def read(name):return json.loads((H/name).read_text(encoding='utf-8-sig'))
def write(name,value):
    p=H/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

def decide(snapshot,estimated_peak_bytes,factor=1.3,mode='runtime'):
    assert math.isfinite(estimated_peak_bytes) and estimated_peak_bytes>0,'MISSING_OR_INVALID_TASK_ESTIMATE'
    assert factor==1.3 and mode in ('startup','runtime'),'UNAPPROVED_POLICY'
    keys=('physical_available_bytes','commit_available_bytes','private_bytes','RSS_bytes')
    assert all(math.isfinite(snapshot[k]) and snapshot[k]>=0 for k in keys)
    physical_need=estimated_peak_bytes if mode=='startup' else max(estimated_peak_bytes-snapshot['RSS_bytes'],0)
    commit_need=estimated_peak_bytes if mode=='startup' else max(estimated_peak_bytes-snapshot['private_bytes'],0)
    physical_threshold=factor*physical_need;commit_threshold=factor*commit_need;process_limit=factor*estimated_peak_bytes
    reasons=[]
    if snapshot['physical_available_bytes']<physical_threshold:reasons.append('AVAILABLE_PHYSICAL_LT_1P3_ESTIMATED_REMAINING')
    if snapshot['commit_available_bytes']<commit_threshold:reasons.append('AVAILABLE_COMMIT_LT_1P3_ESTIMATED_REMAINING')
    if max(snapshot['private_bytes'],snapshot['RSS_bytes'])>process_limit:reasons.append('TASK_EXCEEDS_1P3_ESTIMATED_TOTAL_PEAK')
    return dict(status='PASS' if not reasons else 'STOP_RESOURCE',mode=mode,factor=factor,estimated_total_peak_bytes=estimated_peak_bytes,estimated_remaining_physical_bytes=physical_need,estimated_remaining_commit_bytes=commit_need,required_available_physical_bytes=physical_threshold,required_available_commit_bytes=commit_threshold,dynamic_process_limit_bytes=process_limit,reasons=reasons,snapshot=snapshot,fixed6GiB_rule=False)

def snapshot():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    return dict(at=datetime.now().astimezone().isoformat(),physical_available_bytes=int(m.avail_phys),commit_available_bytes=int(m.avail_page),private_bytes=int(p.private),RSS_bytes=int(p.rss),peak_private_bytes=int(p.peak_pagefile),peak_RSS_bytes=int(p.peak_wset))

def guard(mode='runtime'):
    config=read('resource_policy.json');estimate=read(config['estimate_file']);current=snapshot();result=decide(current,estimate['estimated_total_peak_bytes'],config['safety_factor'],mode)
    RESOURCE['checks']+=1;RESOURCE['peak_private_bytes']=max(RESOURCE['peak_private_bytes'],current['peak_private_bytes']);RESOURCE['peak_RSS_bytes']=max(RESOURCE['peak_RSS_bytes'],current['peak_RSS_bytes'])
    for key,sk in (('min_available_physical_bytes','physical_available_bytes'),('min_available_commit_bytes','commit_available_bytes')):RESOURCE[key]=current[sk] if RESOURCE[key] is None else min(RESOURCE[key],current[sk])
    if result['status']!='PASS':
        result['resource']=dict(RESOURCE);write('RESOURCE_FAILURE.json',result);raise MemoryError('STOP_RESOURCE: task-relative1.3 '+','.join(result['reasons']))
    return result
