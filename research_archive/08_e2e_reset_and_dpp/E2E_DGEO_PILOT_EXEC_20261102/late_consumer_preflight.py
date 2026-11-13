"""Real saved128 consumers and synthetic common-support denominator checks, CPU only."""
import ast,time
from geo_common import *
from geo_extra_review import common_recall,extra_math_fixture,quantile_checked
from compare_geo import summarize_pair,pilot_screen
from causal_dataset import motion
import numpy as np


def main():
    start=time.perf_counter()
    for p in RUN.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    extra_math_fixture();reports={}
    for arm in ARMS:
        root=RUN/'preflight'/arm/'saved128';done=read_json(root/'done.json');verify(done['files'])
        p=dict(np.load(root/'predictions.npz'));e=dict(np.load(root/'window_metrics.npz'));groups=read_json(root/'metrics.json')
        pair=summarize_pair(e,e,groups['overall'],groups['overall'],p,p,np.ones(len(p['ids']),bool))
        if pair['rescue'] or pair['lost'] or pair['coverage_delta']:raise ValueError('IDENTITY_FALSE_EXCHANGE')
        if pilot_screen(groups,groups)['continuous_ADE_pass']:raise ValueError('IDENTITY_FALSE_GAIN')
        common=common_recall({'B0':dict(p,**e),'same':dict(p,**e)})
        if common['overall']['recall']['B0']!=common['overall']['recall']['same']:raise ValueError('COMMON_SELF_MISMATCH')
        m=motion(np.broadcast_to(p['past'][:,None].astype(np.float64),(128,20,8,2)),p['paths'].astype(np.float64))
        checks=0
        for key,value in m.items():
            for q in (.95,.99):quantile_checked(value,q);checks+=1
        reports[arm]=dict(actual_saved_windows=128,self_exchange_no_false_gain=True,common_supported_windows=common['overall']['n'],actual_unconditional_quantile_checks=checks)
    write_json(RUN/'LATE_CONSUMER_PREFLIGHT.json',dict(status='PASS_ACTUAL_SAVED_COMMON_SUPPORT_TAIL_AND_READER',reports=reports,
        elapsed_seconds=time.perf_counter()-start,no_training_or_model_forward=True))
    print('GEO_LATE_CONSUMER_PASS',flush=True)


if __name__=='__main__':main()
