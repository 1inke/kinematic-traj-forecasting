"""Task-estimated1.3 memory guard; never a permanent GiB cap."""
import ctypes,os
import psutil

def decide(snapshot,estimated_peak_bytes,mode='runtime'):
    if estimated_peak_bytes<=0 or mode not in ('startup','runtime'):raise ValueError('Explicit positive estimate and mode required')
    e=estimated_peak_bytes;limit=1.3*e;s=snapshot
    physical=1.3*(e if mode=='startup' else max(e-s['rss'],0));commit=1.3*(e if mode=='startup' else max(e-s['private'],0))
    ok=s['physical']>=physical and s['commit']>=commit and max(s['rss'],s['private'],s['peak_rss'],s['peak_private'])<=limit
    return dict(status='PASS' if ok else 'STOP_RESOURCE',estimate=e,factor=1.3,required_physical=physical,required_commit=commit,retained_peak_limit=limit,snapshot=s.copy())

def snapshot():
    p=psutil.Process().memory_info()
    if os.name!='nt':raise RuntimeError('Live commit guard implemented for Windows; provide independently verified platform adapter')
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('GlobalMemoryStatusEx failed')
    return dict(physical=int(m.avail_phys),commit=int(m.avail_page),rss=int(p.rss),private=int(p.private),peak_rss=int(p.peak_wset),peak_private=int(p.peak_pagefile))

class Guard:
    def __init__(self,estimate):self.estimate=estimate;self.observations=[]
    def check(self,mode='runtime'):
        value=decide(snapshot(),self.estimate,mode);self.observations.append(value)
        if len(self.observations)>128:self.observations=self.observations[-128:]
        if value['status']!='PASS':raise MemoryError(str(value))
        return value
