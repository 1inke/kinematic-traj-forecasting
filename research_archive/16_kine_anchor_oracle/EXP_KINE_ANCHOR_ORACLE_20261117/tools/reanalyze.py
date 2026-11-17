"""Rebuild saved-error summaries. Does not re-fit templates or reconstruct raw geometry."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'snapshot'
FOLDS=('eth','hotel','univ','zara01','zara02')
MAIN=('P','P_AUG','S','K')
MANEUVERS=('stationary','start','stop','turn_and_speed','turn','acceleration','deceleration','steady')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def same(a,b,path=''):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float,np.number)) and not isinstance(a,(bool,np.bool_)):
        np.testing.assert_allclose(a,b,atol=3e-11,rtol=3e-11,err_msg=path)
    else:assert a==b,(path,a,b)
def masks(w):
    labels=w['labels'];g={'overall':np.ones(len(labels),bool)}
    g.update({'maneuver:'+m:labels==m for m in MANEUVERS})
    g['decel_stop']=np.isin(labels,['deceleration','stop'])
    g.update({'speed:'+name:w['speed_bin']==i for i,name in enumerate(('low','middle','high'))})
    g.update({'change:'+name:w['change']==i for i,name in enumerate(('stable','transition','change'))})
    return g
def block(a,f,mask):
    n=int(mask.sum())
    if not n:return dict(n=0)
    return dict(n=n,small=n<20,minADE=float(a[mask].mean()),minFDE=float(f[mask].mean()),FDE_p90=float(np.quantile(f[mask],.9)),reach_FDE_le_0p4=float((f[mask]<=.4).mean()))
def main():
    manifest=read(ROOT/'ARTIFACT_MANIFEST.json')['files']
    for rel,h in manifest.items():assert sha(ROOT/rel)==h,rel
    result=read(SNAP/'RESULT.json');templates=arrays(SNAP/'TEMPLATES.npz');assert len(templates)==150
    spec=importlib.util.spec_from_file_location('fixed_kine_decide',ROOT/'control/design_original/code/kine_decide.py')
    kd=importlib.util.module_from_spec(spec);spec.loader.exec_module(kd)
    summary=dict(folds={},pooled_test={});recomputed={};test_data=[];populations=0;windows=0;cells=0
    for fold in FOLDS:
        summary['folds'][fold]={};recomputed[fold]={}
        for split in ('val','test'):
            w=arrays(SNAP/'analysis_inputs'/f'{fold}_{split}_seed_metrics.npz');old=arrays(SNAP/f'PER_WINDOW_{fold}_{split}.npz');g=masks(w);reports={};n=len(w['ids'])
            for name in ('ids','labels','speed_bin','change','v_rec'):np.testing.assert_array_equal(old[name],w[name]) if name!='v_rec' else np.testing.assert_allclose(old[name],w[name],atol=3e-11,rtol=3e-11)
            assert len(np.unique(w['ids']))==n
            for arm in result['folds'][fold]['splits'][split]:
                aa=w[arm+'__minADE'];ff=w[arm+'__minFDE'];expected_seed_count=1 if arm.startswith('REF_') and arm!='REF_P_ORACLE_SCALE' else 3
                assert aa.shape==ff.shape==(expected_seed_count,n)
                assert np.isfinite(aa).all() and np.isfinite(ff).all() and (aa>=0).all() and (ff>=0).all()
                reports[arm]=[{name:block(a,f,mask) for name,mask in g.items()} for a,f in zip(aa,ff)]
                same(reports[arm],result['folds'][fold]['splits'][split][arm],fold+'/'+split+'/'+arm)
                cells+=len(g)*expected_seed_count
                if arm in MAIN:
                    np.testing.assert_allclose(aa.mean(0),old[arm+'_minADE'],atol=3e-11,rtol=3e-11)
                    np.testing.assert_allclose(ff.mean(0),old[arm+'_minFDE'],atol=3e-11,rtol=3e-11)
            summary['folds'][fold][split]={arm:dict(minADE=[b['overall']['minADE'] for b in ps],minFDE=[b['overall']['minFDE'] for b in ps]) for arm,ps in reports.items() if len(ps)==3}
            recomputed[fold][split]=reports
            if split=='test':test_data.append(w)
            populations+=1;windows+=n
        # The human-readable template table is derivable from saved K parameters.
        table=templates[fold+'__K__0'];rho=np.maximum(np.exp(table[:,:3])-.05,0);phi=np.degrees(table[:,3:])
        reconstructed=[dict(rho=[round(float(v),3) for v in rho[j]],phi_deg=[round(float(v),1) for v in phi[j]]) for j in np.argsort(rho[:,2])]
        same(reconstructed,result['folds'][fold]['templates']['K_seed0'],fold+'/template_table')
    for arm in MAIN:
        summary['pooled_test'][arm]=dict(decel_stop={})
        for metric in ('minADE','minFDE'):
            values=[]
            for seed in range(3):
                b=[recomputed[f]['test'][arm][seed]['decel_stop'] for f in FOLDS];n=sum(x['n'] for x in b)
                values.append(sum(x.get(metric,0)*x['n'] for x in b)/n)
            summary['pooled_test'][arm]['decel_stop'][metric]=values
    same(summary,result['summary'],'summary')
    same(kd.decide(summary),result['decision'],'decision')
    pooled={}
    for group in ('decel_stop','steady','speed_high'):
        pooled[group]={}
        for arm in MAIN:
            aa=[];ff=[]
            for w in test_data:
                mask=np.isin(w['labels'],['deceleration','stop']) if group=='decel_stop' else w['labels']=='steady' if group=='steady' else w['speed_bin']==2
                aa.append(w[arm+'__minADE'].mean(0)[mask]);ff.append(w[arm+'__minFDE'].mean(0)[mask])
            a=np.concatenate(aa);f=np.concatenate(ff)
            pooled[group][arm]=dict(n=len(a),minADE=float(a.mean()) if len(a) else None,minFDE=float(f.mean()) if len(f) else None)
    same(pooled,result['pooled_test_seedmean'],'pooled_test_seedmean')
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='PASS',manifest_files=len(manifest),populations=populations,windows=windows,group_seed_blocks=cells,template_arrays=len(templates),all_main_sensitivity_and_reference_groups='MATCH',seedwise_p90_and_reach='MATCH',original_seed_mean_arrays='MATCH',C1_to_C6_and_attribution='MATCH',verdict=result['decision']['verdict'],scope='Saved-error reaggregation and template-table reconstruction; no raw-coordinate, label or clustering reconstruction.',training=False,inference=False,GPU_used=False),ensure_ascii=False))

if __name__=='__main__':main()
