"""Independent scalar-window equations against saved source predictions and summaries."""
from pathlib import Path
import json,sys,hashlib,time,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def eq(a,b):np.testing.assert_allclose(a,b,atol=3e-11,rtol=3e-11)
def checkfiles(p):
    d=read(p)
    for f,h in d['files'].items():assert sha(f)==h,f
    return len(d['files'])
started=time.monotonic();source_n=checkfiles(ROOT/'SOURCE_MANIFEST.json');output_n=checkfiles(ROOT/'FINAL_OUTPUT_SEAL.json')
cfg=read(ROOT/'CONFIG.json');result=read(ROOT/'DECEL_DIAG.json')
sys.path.insert(0,cfg['fivefold_code'])
from fivefold_data import load_split
truth={'ETH':load_split(cfg['data_dir'],'eth','test'),'C0_dev':load(cfg['C0_dev'])}
spec=importlib.util.spec_from_file_location('labels',cfg['maneuver_source']);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
verified=[];conditional=[]
for item in cfg['populations']:
    name,model=item['dataset'],item['model'];d=truth[name];pred=load(item['prediction']);w=load(result['per_window_files'][name+'/'+model])
    lookup={v:i for i,v in enumerate(pred['ids'].tolist())};order=np.array([lookup[v] for v in d['ids'].tolist()]);x=pred['paths'][order].astype(float);z=pred['logits'][order].astype(float)
    np.testing.assert_array_equal(w['ids'],d['ids']);np.testing.assert_array_equal(w['maneuver'],helper.maneuver_labels(d['past'],d['future']))
    p=d['past'][:,0].astype(float);y=d['future'].astype(float);n=len(p)
    for i in range(n):
        obs=np.array([np.hypot(*(p[i,j+1]-p[i,j]))/.4 for j in range(7)])
        vobs=obs.mean();base=max(vobs,.1);vec=np.mean(np.diff(p[i,-4:],axis=0),axis=0);norm=np.hypot(*vec);u=vec/max(norm,1e-12);normal=np.array([-u[1],u[0]])
        def ratio(path):return sum(np.hypot(*(b-a)) for a,b in zip(np.vstack([p[i,-1],path])[:-1],path))/(12*.4*base)
        rg=ratio(y[i]);rk=np.array([ratio(path) for path in x[i]]);probs=np.exp(z[i]-max(z[i]));probs/=sum(probs)
        eq(w['r_g'][i],rg);eq(w['candidate_r'][i],rk);eq(w['P_dec'][i],sum(probs[rk<=.8]));eq(w['observed_speed'][i],vobs)
        eq(w['history_ratio'][i],obs[-3:].mean()/max(obs[:4].mean(),.1));assert w['direction_valid'][i]==(norm>.03)
        assert w['n_decel'][i]==sum(rk<=.8);assert w['speed_ratio_reachable'][i]==any(abs(rk-rg)<=.15)
        ranks=sorted(range(20),key=lambda k:(-z[i,k],k));top=ranks[0]
        for t in (6,12):
            s=f'_t{t}';delta=x[i,:,t-1]-y[i,t-1];prog=delta@u;lat=delta@normal;distance=np.hypot(delta[:,0],delta[:,1]);best=int(np.argmin(distance))
            eq(w['candidate_prog_error'+s][i],prog);eq(w['candidate_lat_error'+s][i],lat)
            eq(w['best_prog_error'+s][i],prog[best]);eq(w['best_lat_error'+s][i],lat[best]);eq(w['best_displacement_error'+s][i],distance[best]);eq(w['top1_prog_error'+s][i],prog[top])
            assert w['best_rank'+s][i]==ranks.index(best)+1
            assert w['envelope_covered'+s][i]==(min(prog)<=0<=max(prog))
            assert w['all_overshoot'+s][i]==(min(prog)>0)
            assert w['all_undershoot'+s][i]==(max(prog)<0)
            assert w['longitudinal_reachable'+s][i]==(min(abs(prog))<=.4)
            assert w['top1_overshoot'+s][i]==(prog[top]>.4)
    report=result['datasets'][name][model]
    for cohort,c in report['cohorts'].items():
        subset=np.ones(n,bool) if cohort=='all' else np.isin(w['maneuver'],['deceleration','stop','turn_and_speed']) if cohort=='state_change' else np.isin(w['maneuver'],['steady','acceleration']) if cohort=='controls' else w['maneuver']==cohort
        assert c['total_windows']==sum(subset)
        groups=[(c['all_history'],subset&w['direction_valid'])]+[(c['by_history'][h],subset&w['direction_valid']&(w['history_bin']==j)) for j,h in enumerate(('decelerating','steady','accelerating'))]
        for b,mask in groups:
            assert b['n']==sum(mask)
            for cb in b['calibration']:
                m=mask&(w['P_dec']>=cb['lower'])&((w['P_dec']<cb['upper']) if cb['bin']<4 else w['P_dec']<=1)
                assert cb['n']==sum(m)
                if m.any():eq(cb['mean_P_dec'],w['P_dec'][m].mean());eq(cb['actual_deceleration_rate'],w['true_deceleration'][m].mean())
            if not mask.any():continue
            for t in (6,12):
                b0=b[f't{t}']
                for raw,field in [('envelope_covered','envelope_coverage_rate'),('all_overshoot','all_overshoot_rate'),('all_undershoot','all_undershoot_rate'),('longitudinal_reachable','longitudinal_reachable_rate'),('top1_overshoot','top1_overshoot_rate')]:eq(b0[field],w[raw+f'_t{t}'][mask].mean())
                for raw in ('best_prog_error','best_lat_error','top1_prog_error'):
                    a=w[raw+f'_t{t}'][mask];v=b0[raw]
                    eq(v['signed_mean'],a.mean());eq(v['absolute_mean'],abs(a).mean());eq(v['absolute_p90'],np.percentile(abs(a),90))
    m=w['direction_valid']&(w['maneuver']=='deceleration');reach=w['longitudinal_reachable_t12'];over=w['top1_overshoot_t12'];decision=report['decision']
    A=(w['all_overshoot_t12'][m].mean()>=.3) or reach[m].mean()<.6;B=reach[m].mean()>=.7 and over[m].mean()>=.5
    assert decision['A']==A and decision['B']==B
    hd=m&(w['history_bin']==0);hs=m&(w['history_bin']==1);allhd=w['direction_valid']&(w['history_bin']==0)
    rate=hd.any() and hs.any() and over[hd].mean()>=over[hs].mean();under=allhd.any() and w['true_deceleration'][allhd].mean()-w['P_dec'][allhd].mean()>=.2
    calibrated=all(abs(c['gap_actual_minus_predicted'])<.1 for c in report['cohorts']['all']['all_history']['calibration'] if c['n'])
    steadyshare=sum(over&m&(w['history_bin']==1))/max(sum(over&m),1)
    assert decision['B1']==bool(B and (rate or under));assert decision['B2']==bool(B and steadyshare>.5 and calibrated)
    conditional.append(dict(dataset=name,model=model,n_deceleration=int(sum(m)),n_reachable=int(sum(m&reach)),n_reachable_but_top1_overshoot=int(sum(m&reach&over)),top1_overshoot_given_reachable=float(over[m&reach].mean())))
    verified.append(dict(dataset=name,model=model,windows=n,primitive_checks='ALL_WINDOWS_ALL_CANDIDATES_TWO_HORIZONS',all_cohort_history_aggregates='PASS',decision='PASS'))
assert 'torch' not in sys.modules
value=dict(status='PASS',source_files=source_n,sealed_output_files=output_n,populations=verified,total_windows=sum(v['windows'] for v in verified),conditional_selection=conditional,seconds=time.monotonic()-started,training=False,inference=False,GPU_used=False,script_sha256=sha(__file__))
(ROOT/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(value,ensure_ascii=False))
