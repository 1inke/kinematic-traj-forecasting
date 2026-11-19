"""Verify the archived saved-metric evidence without models, coordinates or Torch."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT/'snapshot'
METRICS = ('minADE','minFDE','top1_ADE','top1_FDE')
COMMON = (*METRICS,'covered','top1_good')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def close(a,b):
    np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11)


def masks(w, labels):
    g = w['group']
    values = dict(overall=np.ones(len(g),bool),stable=g==0,transition=g==1,change=g==2,
                  turn_any=np.isin(labels,['turn','turn_and_speed']),decel_stop=np.isin(labels,['deceleration','stop']))
    values.update({k:labels==k for k in set(labels)})
    return values


def subset(w,prefix):
    prefix += '__'
    return {k[len(prefix):]:v for k,v in w.items() if k.startswith(prefix)}


def check_summary(w,report,subsets):
    for name,a in report.items():
        group = subsets[name]
        assert a['n']==int(group.sum())
        for key in set(a)&set(w)-{'selected_slot','retired_slots'}:
            close(w[key][group].mean(),a[key])
        assert a['no_joint_good_candidate_n']==int((~w['covered'][group]).sum())
        assert a['good_exists_top1_missed_n']==int((w['covered']&~w['top1_good'])[group].sum())


def check_pair(ref,new,group,reported,mode):
    gain = reported['relative_gains' if mode=='pilot' else 'improvement']
    for k,g in gain.items():
        close(1-new[k][group].mean()/ref[k][group].mean(),g)
    for key in ('covered','top1_good') if mode=='history' else ('covered',):
        previous,current = ref[key][group],new[key][group]
        rescue,lost = int((current&~previous).sum()),int((previous&~current).sum())
        if mode=='pilot':
            assert (rescue,lost)==(reported['rescued'],reported['lost'])
            close(reported['coverage_delta_pp'],100*(rescue-lost)/len(previous))
        else:
            assert (rescue,lost)==(reported[key]['rescue'],reported[key]['lost'])
            close(reported[key]['difference_pp'],100*(rescue-lost)/len(previous))


def protection(comparison, gain_key, error_key, motion_key, new_motion, base_motion):
    checks = [('overall','top1_FDE',.02),('change','minFDE',.01),
              ('turn_any','minFDE',.01),('decel_stop','minFDE',.01)]
    flags = [(g,k) for g,k,t in checks if comparison['groups'][g][gain_key][k]<-t]
    assert flags==[(v['group'],v['metric']) for v in comparison[error_key]]
    recalculated=[]
    undefined=[]
    for group in ('overall','stable','transition','change'):
        for port, ref in base_motion[group].items():
            new=new_motion[group][port]
            if port!='Top1':
                for key,value in new['abs_log_ratio'].items():
                    b=ref['abs_log_ratio'][key]
                    if value is None or b is None:
                        undefined.append((group,port,key))
                    elif value-b>.05:
                        recalculated.append((group,port,key,'abs_log_ratio',value-b))
            for key,value in new['exceedance_rate'].items():
                diff=value-ref['exceedance_rate'][key]
                if diff>.01:
                    recalculated.append((group,port,key,'tail_rate',diff))
    expected = [(r['group'],r['port'],r['metric'],r['kind'],r['increase']) for r in comparison[motion_key]]
    assert sorted(expected)==sorted(recalculated)
    assert comparison['protection_pass']==(not flags and not recalculated and not undefined)


def diag():
    root=SNAP/'PREC_DIAG'
    r=read(root/'RESULT.json')
    count=0
    for arm,seeds in r['families'].items():
        for seed,row in seeds.items():
            w=arrays(root/'windows'/f'{arm}_s{seed}.npz')
            g=masks(w,w['maneuver'])
            b=subset(w,'base')
            check_summary(b,row['baseline'],g)
            count+=1
            for name,c in row['options'].items():
                x=subset(w,name)
                check_summary(x,c['groups'],g)
                for group,out in c['groups'].items():
                    m=g[group]
                    for key in ('minADE','minFDE','top1_FDE'):
                        close(1-x[key][m].mean()/b[key][m].mean(),out[key+'_relative_gain'])
                    close(100*(x['covered'][m].mean()-b['covered'][m].mean()),out['coverage_gain_pp'])
                    if c['K']==20:
                        for k in ('minADE','minFDE'):
                            delta=x[k+'_net_gain_m'][m]
                            close(b[k][m]-x[k][m],delta)
                            close(delta,x[k+'_add_after_delete_gain_m'][m]-x[k+'_delete_harm_m'][m])
                            close(delta[delta>0].sum(),out[k+'_positive_gain_sum_m'])
                            close(-delta[delta<0].sum(),out[k+'_negative_gain_sum_m'])
                            for f in ('add_after_delete_gain_m','delete_harm_m','lost_vs_union_m'):
                                close(x[k+'_'+f][m].sum(),out[k+'_'+f+'_sum'])
                        assert int((x['covered']&~b['covered'])[m].sum())==out['coverage_gained_n']
                        assert int((b['covered']&~x['covered'])[m].sum())==out['coverage_lost_n']
                count+=1
        for name,panel in r['panel'][arm].items():
            for g,group in panel['groups'].items():
                for k,value in group.items():
                    close(np.mean([row['options'][name]['groups'][g][k] for row in seeds.values()]),value)
    assert count==80 and r['panel']['B0']['lowest_logit_CV']['gain_screen']
    # Geometry feasibility and gradient checks are preserved as local receipts.
    assert read(root/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    return count


def pilot():
    root=SNAP/'PREC_PILOT'
    r=read(root/'RESULT.json')
    windows={}
    for name,model in r['models'].items():
        w=arrays(root/'windows'/f'{name}.npz')
        g=masks(w,w['maneuver'])
        check_summary({k:v for k,v in w.items() if k not in ('ids','group','maneuver')},model['groups'],g)
        windows[name]=w
    for key,comp in r['comparisons'].items():
        arm,ref=key.split('_vs_')
        a,b=windows[arm],windows[ref]
        np.testing.assert_array_equal(a['ids'],b['ids'])
        for group,report in comp['groups'].items():
            check_pair(b,a,g[group],report,'pilot')
        protection(comp,'relative_gains','error_failures','motion_failures',r['models'][arm]['motion'],r['models'][ref]['motion'])
    c=r['comparisons']['Prior_self_vs_E3_self']['groups']['overall']['relative_gains']
    gain=c['minADE']>=.015 and c['minFDE']>=.015
    decision=r['outcome']
    protections=all(r['comparisons'][v]['protection_pass'] for v in decision['compared_for_protection'])
    assert gain==decision['screening_gain_pass']==False
    assert protections==decision['protection_pass']==False
    assert decision['recommend_more_seeds']==(gain and protections)
    orders=read(root/'CONTROLS.json')['controls']['E3_self']['order_hashes']
    for arm in ('Prior','Prior_self'):
        receipts=[read(root/'runs'/arm/f'ckpt_epoch{i:03d}.json') for i in range(1,33)]
        assert [x['order_hash'] for x in receipts]==orders
        assert [x['updates'] for x in receipts]==[340*i for i in range(1,33)]
    return windows['E3_self']


def history(ref):
    root=SNAP/'PREC_HISTORY_DIAG'
    r=read(root/'RESULT.json')
    count=0
    for arm,options in r['models'].items():
        w=arrays(root/f'{arm}_windows.npz')
        np.testing.assert_array_equal(w['ids'],ref['ids'])
        np.testing.assert_array_equal(w['group'],ref['group'])
        g=masks(w,ref['maneuver'])
        g.update(obs_stationary=w['stationary'],obs_steady=w['steady'],obs_other=~w['stable_history'])
        np.testing.assert_array_equal(w['stable_history'],w['stationary']|w['steady'])
        base=subset(w,'original')
        for name,out in options.items():
            x=subset(w,name)
            check_summary(x,out['groups'],g)
            for refkey,b,meta in [('vs_original',base,options['original']),('vs_E3_self',ref,r['E3_self'])]:
                comp=out[refkey]
                for group,report in comp['groups'].items():
                    check_pair(b,x,g[group],report,'history')
                protection(comp,'improvement','error_flags','motion_flags',out['motion'],meta['motion'])
            assert out['score_changed_n']==int((x['selected_slot']!=base['selected_slot']).sum())
            count+=1
    assert count==14
    valid=r['models']['Prior_self']['CV_if_stationary']
    assert valid['versus_E3_self_dual_gain_pass'] and not valid['vs_E3_self']['protection_pass']
    assert valid['vs_original']['groups']['overall']['covered']['lost']==0
    return count


def main():
    manifest=read(ROOT/'ARTIFACT_MANIFEST.json')['files']
    for name,expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    counts=dict(PREC_DIAG=diag(),PREC_PILOT=6)
    counts['PREC_HISTORY_DIAG']=history(pilot())
    print(json.dumps(dict(status='PASS',manifest_entries=len(manifest),evaluation_populations=counts,
          dev_window_records=sum(counts.values())*4753,paired_epochs=64,
          protects_from_saved_motion_summaries=True,
          scientific_outcome='PILOT_SCREEN_NOT_PASSED',new_inference=False,new_training=False)))


if __name__=='__main__':
    main()
