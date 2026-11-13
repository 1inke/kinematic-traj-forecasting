"""Synthetic branches plus real saved128 dual-screen/factorial consumers, not model fits."""
from geo_common import *
from geo_extra_review import factorial_contrast,extra_screen
from compare_geo import pilot_screen,GROUPS,TAILS
import copy

def run():
    x=factorial_contrast(1.,3.,4.,10.)
    if x['A_at_native']!=2 or x['A_at_Taylor']!=6 or x['B_at_no_geo']!=3 or x['B_at_maneuver']!=7 or x['interaction']!=4:raise ValueError('FACTORIAL_CONTRAST')
    risk=[dict(group=g,metric=k,status='ESTIMABLE' if k!='jerk' else 'NOT_ESTIMABLE_NO_FLOOR',rate_pass=True,magnitude_pass=True if k!='jerk' else None) for g in GROUPS for k in TAILS]
    s=dict(continuous_ADE_pass=True,FDE_noninferiority_pass=True,full_pilot_screen=False,tail_comparisons=risk)
    m={g:dict(coverage=.5,effective_good=.6) for g in GROUPS}
    rows=[dict(arm='fixture',split='dev',metric='jerk',new_exceed=0) for g in GROUPS]
    q=extra_screen(s,rows,'fixture',m,m)
    if q['strict_literal_C0_screen'] or not q['estimable_16_plus_zero_jerk_pass'] or not q['strict_zero_jerk_new_exceed_eq0']:raise ValueError('SEPARATE_ZERO_REFERENCE_SCREENS')
    rows[0]['new_exceed']=1
    q=extra_screen(s,rows,'fixture',m,m)
    if not q['estimable_16_plus_zero_jerk_pass'] or q['strict_zero_jerk_new_exceed_eq0']:raise ValueError('RATE_TOLERANCE_NOT_ZERO')
    risk[0]['magnitude_pass']=False
    if extra_screen(s,rows,'fixture',m,m)['estimable_16_plus_zero_jerk_pass']:raise ValueError('FAILURE_HIDDEN')
    real={}
    for arm in ARMS:
        groups=read_json(RUN/'preflight'/arm/'saved128/metrics.json')
        selfscreen=pilot_screen(groups,groups)
        if selfscreen['continuous_ADE_pass']:raise ValueError('SELF_FALSE_ACCURACY_GAIN')
        q=extra_screen(selfscreen,[],'fixture',groups,groups)
        if q['estimable_16_plus_zero_jerk_pass'] or q['strict_zero_jerk_new_exceed_eq0']:raise ValueError('EMPTY_ROWS_FALSE_PASS')
        real[arm]=dict(no_self_gain=True,missing_zero_support_not_pass=True)
    return dict(status='PASS_SYNTHETIC_AND_ACTUAL128_FACTORIAL_DUALSCREEN',real=real,synthetic_zero_relative_not_pass=True,no_training=True)

