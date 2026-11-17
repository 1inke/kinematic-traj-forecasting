"""CPU reaggregation from archived derived metrics; no raw-coordinate recomputation."""
from pathlib import Path
import hashlib,json,sys,importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'snapshot'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b,path=''):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for key in a:same(a[key],b[key],path+'/'+key)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,atol=2e-12,rtol=2e-12,err_msg=path)
    else:assert a==b,(path,a,b)


def main():
    manifest=read(ROOT/'ARTIFACT_MANIFEST.json')['files']
    for rel,digest in manifest.items():assert sha(ROOT/rel)==digest,rel
    spec=importlib.util.spec_from_file_location('frozen_decel_metrics',SNAP/'code/decel_metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    result=read(SNAP/'DECEL_DIAG.json');receipt=read(SNAP/'INDEPENDENT_VERIFICATION.json');conditional=[];groups=0;records=0
    for dataset,models in result['datasets'].items():
        for model,report in models.items():
            path=SNAP/'populations'/dataset/model/'decel_window_metrics.npz'
            with np.load(path,allow_pickle=False) as archive:w={k:archive[k] for k in archive.files}
            n=len(w['ids']);assert n==report['n_total'] and len(np.unique(w['ids']))==n
            assert w['candidate_r'].shape==(n,20)
            np.testing.assert_array_equal(w['direction_valid'],w['mean_displacement_length']>.03)
            np.testing.assert_array_equal(w['direction_valid_previous_total'],w['mean_displacement_length']*3>.03)
            np.testing.assert_array_equal(w['true_deceleration'],w['r_g']<=.8)
            np.testing.assert_array_equal(w['candidate_decel'],w['candidate_r']<=.8)
            np.testing.assert_array_equal(w['n_decel'],w['candidate_decel'].sum(-1))
            np.testing.assert_allclose(w['candidate_probability'].sum(-1),1,atol=2e-12)
            np.testing.assert_allclose(w['P_dec'],np.clip((w['candidate_probability']*w['candidate_decel']).sum(-1),0,1),atol=2e-12)
            np.testing.assert_array_equal(w['speed_ratio_reachable'],(abs(w['candidate_r']-w['r_g'][:,None])<=.15).any(-1))
            np.testing.assert_array_equal(w['history_bin'],np.where(w['history_ratio']<=.85,0,np.where(w['history_ratio']>=1.15,2,1)))
            for name,func in [('candidate_r_min',np.min),('candidate_r_max',np.max),('candidate_r_mean',np.mean)]:
                np.testing.assert_allclose(w[name],func(w['candidate_r'],axis=-1),atol=2e-12)
            rows=np.arange(n);valid=w['direction_valid']
            for t in (6,12):
                s=f'_t{t}';prog=w['candidate_prog_error'+s];lat=w['candidate_lat_error'+s]
                best=w['best_index'+s];top=w['top1_index']
                np.testing.assert_allclose(w['best_prog_error'+s],prog[rows,best],atol=2e-12)
                np.testing.assert_allclose(w['best_lat_error'+s],lat[rows,best],atol=2e-12)
                np.testing.assert_allclose(w['top1_prog_error'+s],prog[rows,top],atol=2e-12)
                np.testing.assert_array_equal(w['top1_overshoot'+s],prog[rows,top]>.4)
                np.testing.assert_array_equal(w['longitudinal_reachable'+s],abs(prog).min(-1)<=.4)
                np.testing.assert_array_equal(w['envelope_covered'+s],(prog.min(-1)<=0)&(prog.max(-1)>=0))
                np.testing.assert_allclose(w['best_displacement_error'+s][valid],np.hypot(prog,lat).min(-1)[valid],atol=2e-11,rtol=2e-11)
            same(metrics.summarize(w,w['maneuver']),report['cohorts'],dataset+'/'+model+'/main')
            same(metrics.decide(w,w['maneuver']),report['decision'],dataset+'/'+model+'/decision')
            previous=report['sensitivity_previous_total_direction']
            same(metrics.summarize(w,w['maneuver'],'direction_valid_previous_total'),previous['cohorts'],dataset+'/'+model+'/sensitivity')
            same(metrics.decide(w,w['maneuver'],'direction_valid_previous_total'),previous['decision'],dataset+'/'+model+'/sensitivity_decision')
            mask=valid&(w['maneuver']=='deceleration');reach=w['longitudinal_reachable_t12'];over=w['top1_overshoot_t12']
            conditional.append(dict(dataset=dataset,model=model,n_deceleration=int(sum(mask)),n_reachable=int(sum(mask&reach)),n_reachable_but_top1_overshoot=int(sum(mask&reach&over)),top1_overshoot_given_reachable=float(over[mask&reach].mean())))
            groups+=1;records+=n
    same(conditional,receipt['conditional_selection'],'conditional_selection')
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='PASS',manifest_files=len(manifest),populations=groups,model_window_records=records,all_maneuver_history_metrics='MATCH',previous_direction_sensitivity='MATCH',all_decisions='MATCH',conditional_selection='MATCH',scope='Reaggregation and derived-field consistency; not raw-coordinate error or label reconstruction.',training=False,inference=False,GPU_used=False),ensure_ascii=False))


if __name__=='__main__':main()
