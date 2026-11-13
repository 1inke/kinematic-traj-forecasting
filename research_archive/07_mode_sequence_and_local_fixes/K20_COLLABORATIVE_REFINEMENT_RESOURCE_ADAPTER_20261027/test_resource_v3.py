from core import *
from resource_guard_v3 import decide, Guard, snapshot, MIB

def main():
    e=read_json(HERE/'RESOURCE_STAGE_PLAN.json')['GPU_estimate_bytes']; gib=1024**3
    def s(**kw):
        z=dict(physical=14*gib,commit=7*gib,physical_total=32*gib,commit_total=33*gib,commit_limit=40*gib,rss=gib,private=gib,peak_rss=gib,peak_private=gib);z.update(kw);return z
    checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(name)
    v=decide(s(physical=4*gib,commit=4*gib),e,'startup')
    check('below1p3_but_fits_actual_new_allocation_warn_not_false_block',v['status']=='PASS' and v['severity']=='WARNING')
    check('real_admission_shortfall_blocks',decide(s(commit=3*gib),e,'startup')['status']=='STOP_RESOURCE')
    v=decide(s(private=2*gib,rss=2*gib,physical=3*gib,commit=3*gib),e)
    check('runtime_subtracts_already_owned_memory',v['remaining_commit']==e-2*gib and v['status']=='PASS')
    v=decide(s(peak_private=6*gib),e)
    check('historical_peak_not_current_memory_lock',v['status']=='PASS' and v['severity']=='WARNING')
    v=decide(s(private=5*gib,rss=5*gib),e)
    check('estimate_undershoot_healthy_task_warns_without_inflation',v['status']=='PASS' and v['estimate']==e and v['severity']=='WARNING')
    def replay(values,times):
        it=iter(values);t=iter(times);return Guard(e,sampler=lambda:next(it),clock=lambda:next(t))
    g=replay([s(commit=256*MIB),s(),s(commit=256*MIB)],[0,5,10])
    check('transient_pressure_resets_not_failed',all(g.check(force=True)['status']=='PASS' for _ in range(3)))
    g=replay([s(commit=256*MIB)]*3,[0,5,10]);g.check(force=True);g.check(force=True)
    try:g.check(force=True);raise AssertionError('persistent pressure was not stopped')
    except MemoryError:checks.append('persistent10s_three_observations_stop_at_check')
    check('emergency64MiB_stops',decide(s(physical=63*MIB),e)['status']=='STOP_RESOURCE')
    g=replay([s()],[0,1]);g.check();g.check()
    check('runtime_check_throttled_without_sleep_or_extra_sampling',len(g.observations)==1)
    try:decide(s(commit=-1),e);raise AssertionError('bad telemetry accepted')
    except ValueError:checks.append('invalid_telemetry_not_silently_passed')
    live=snapshot();v=decide(live,e,'startup')
    write_json(HERE/'RESOURCE_POLICY_TESTS.json',dict(status='PASS',policy='warning_first_remaining_v3',checks=checks,one_live_snapshot=live,live_admission=v,old_reports_and_science_untouched=True,no_OS_pagefile_user_process_changes=True));print('PASS RESOURCE V3',len(checks),v['severity'],flush=True)
if __name__=='__main__':main()
