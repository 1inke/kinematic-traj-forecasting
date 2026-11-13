"""Warning-first resource admission, authorized by the human on 2026-10-27.

No memory is reserved/locked by this monitor. No OS or pagefile mutations.
The estimate is planning information, not a process kill limit.
"""
import ctypes
import os
import time
import psutil

MIB = 1024**2

def snapshot():
    if os.name != 'nt':
        raise RuntimeError('Windows resource adapter required')
    size = ctypes.c_size_t
    class Performance(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong)] + [(n, size) for n in (
            'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
            'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',
            'KernelNonpaged', 'PageSize')] + [(n, ctypes.c_ulong) for n in (
            'HandleCount', 'ProcessCount', 'ThreadCount')]
    p = Performance(); p.cb = ctypes.sizeof(p)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb):
        raise OSError('GetPerformanceInfo failed; cannot infer available memory')
    m = psutil.Process().memory_info()
    return dict(physical=int(p.PhysicalAvailable*p.PageSize),
                commit=int((p.CommitLimit-p.CommitTotal)*p.PageSize),
                physical_total=int(p.PhysicalTotal*p.PageSize),
                commit_total=int(p.CommitTotal*p.PageSize),
                commit_limit=int(p.CommitLimit*p.PageSize),
                rss=int(m.rss), private=int(m.private),
                peak_rss=int(m.peak_wset), peak_private=int(m.peak_pagefile))

def decide(s, estimate, mode='runtime'):
    if estimate <= 0 or mode not in ('startup', 'runtime'):
        raise ValueError('Positive estimate and startup/runtime mode required')
    if any(s[k] < 0 for k in ('physical','commit','rss','private','peak_rss','peak_private')) or s['physical_total'] <= 0:
        raise ValueError('Invalid memory telemetry')
    # Startup is called in the lightweight coordinator BEFORE allocating the child.
    # Runtime is called inside the task and subtracts memory already allocated.
    rp = estimate if mode == 'startup' else max(estimate-s['rss'], 0)
    rc = estimate if mode == 'startup' else max(estimate-s['private'], 0)
    reserve = max(128*MIB, min(512*MIB, int(.01*s['physical_total'])))
    warnings = []
    if s['physical'] < 1.3*rp: warnings.append('PHYSICAL_PLANNING_MARGIN_BELOW_1P3')
    if s['commit'] < 1.3*rc: warnings.append('COMMIT_PLANNING_MARGIN_BELOW_1P3')
    if max(s['rss'], s['private'], s['peak_rss'], s['peak_private']) > 1.3*estimate:
        warnings.append('ESTIMATE_UNDERSHOOT_REVIEW_NO_AUTOMATIC_KILL_OR_INFLATION')
    admission_short = mode == 'startup' and (s['physical'] < rp+reserve or s['commit'] < rc+reserve)
    pressure = mode == 'runtime' and min(s['physical'], s['commit']) < reserve
    emergency = mode == 'runtime' and min(s['physical'], s['commit']) < 64*MIB
    if pressure: warnings.append('CURRENT_SYSTEM_PRESSURE')
    return dict(status='STOP_RESOURCE' if admission_short or emergency else 'PASS',
                severity='STOP' if admission_short or emergency else ('WARNING' if warnings else 'NORMAL'),
                policy='warning_first_remaining_v3', mode=mode, estimate=estimate,
                factor=1.3, remaining_physical=rp, remaining_commit=rc,
                warning_physical=1.3*rp, warning_commit=1.3*rc,
                admission_physical=rp+reserve, admission_commit=rc+reserve,
                adaptive_system_reserve=reserve, pressure=pressure,
                emergency=emergency, warnings=warnings, snapshot=s.copy())

class Guard:
    def __init__(self, estimate, sampler=snapshot, clock=time.monotonic):
        self.estimate=estimate; self.observations=[]; self.sampler=sampler; self.clock=clock
        self.last_time=None; self.last=None; self.pressure_since=None; self.pressure_count=0
    def check(self, mode='runtime', force=False):
        t=self.clock()
        if mode == 'runtime' and not force and self.last is not None and self.last['mode']=='runtime' and t-self.last_time < 2:
            return self.last
        value=decide(self.sampler(), self.estimate, mode)
        if mode=='runtime' and value['pressure']:
            if self.pressure_since is None: self.pressure_since=t
            self.pressure_count+=1
            # A small transient shortage is a warning, not an experiment failure.
            if self.pressure_count>=3 and t-self.pressure_since>=10:
                value.update(status='STOP_RESOURCE', severity='STOP', sustained_pressure_seconds=t-self.pressure_since)
        else:
            self.pressure_since=None; self.pressure_count=0
        self.last_time=t; self.last=value; self.observations.append(value)
        self.observations=self.observations[-128:]
        if value['status']!='PASS':
            raise MemoryError(str(value))
        return value




