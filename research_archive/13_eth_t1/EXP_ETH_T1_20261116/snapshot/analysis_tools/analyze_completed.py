"""Read-only CPU audit and descriptive diagnosis of the completed ETH-T1 study."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
W=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'adapter'))
from eth_exec_common import read,sha,verify,verify_source,validate_seal,run_dir
sys.path.insert(0,str(W/'qoder_work/FIVEFOLD_B0_EXEC_20261115/code'))
from fivefold_data import load_split,scene_tail_split
from fivefold_metrics import per_window,aggregate


def equal(a,b,label):
    if isinstance(b,dict):
        assert a.keys()==b.keys(),(label,'keys')
        for k,v in b.items():equal(a[k],v,label+'/'+k)
    elif isinstance(b,(float,int)) and not isinstance(b,bool):
        assert np.isclose(a,b,rtol=1e-12,atol=1e-12),(label,a,b)
    else:assert a==b,(label,a,b)


def main():
    verify_source();seal=read(ROOT/'FINAL_OUTPUT_SEAL.json');verify(seal['files'])
    assert seal['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
    result=read(ROOT/'standalone/RESULT.json')
    assert result['status']=='SUCCESS' and read(ROOT/'standalone/WORKER_EXIT.json')['exit_code']==0
    assert not (ROOT/'standalone/FAILURE.json').exists()
    cfg=read(W/'qoder_work/FIVEFOLD_B0_PARALLEL_20261115/PARALLEL_CONFIG.json')
    test=load_split(cfg['data_dir'],'eth','test');train=load_split(cfg['data_dir'],'eth','train')
    index,_=scene_tail_split(train['scene']);assert len(index)==28791
    data={};metrics={};predictions={}
    for arm in ('B0','T1'):
        d=run_dir(arm);validate_seal(d,'EVALUATION_SEAL.json')
        if arm=='T1':validate_seal(d,'TRAIN_SEAL.json')
        with np.load(d/'test_predictions.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
        with np.load(d/'test_window_metrics.npz',allow_pickle=False) as z:v={k:z[k] for k in z.files}
        assert p['paths'].shape==(364,20,12,2) and p['paths'].dtype==np.float32
        assert np.isfinite(p['paths']).all() and np.isfinite(p['logits']).all()
        for k in ('ids','group'):assert np.array_equal(p[k],test[k]) and np.array_equal(v[k],test[k])
        direct=np.sqrt(((p['paths'].astype(np.float64)-test['future'][:,None].astype(np.float64))**2).sum(-1))
        assert np.allclose(v['minADE'],direct.mean(-1).min(1),rtol=1e-12,atol=1e-12)
        assert np.allclose(v['minFDE'],direct[:,:,-1].min(1),rtol=1e-12,atol=1e-12)
        calculated=per_window(p['paths'],p['logits'],test['future'],test['past'][:,0])
        for k,value in calculated.items():assert np.allclose(value,v[k],rtol=1e-12,atol=1e-12),(arm,k)
        m=aggregate(calculated,test['group']);past=test['past'][:,0].astype(np.float64)
        vel1=(p['paths'][:,:,0].astype(np.float64)-past[:,None,-1])/.4
        m['overall']['junction_accel_allK20']=float((np.linalg.norm(vel1-((past[:,-1]-past[:,-2])/.4)[:,None],axis=-1)/.4).mean())
        equal(m,read(d/'result.json')['test'],arm)
        data[arm]=v;metrics[arm]=m;predictions[arm]=p
    main=read(ROOT/'ETH_T1_RESULT.json')
    for arm in ('B0','T1'):equal(metrics[arm]['overall'],main[arm],arm+'/primary')
    gains={k:(metrics['B0']['overall'][k]-metrics['T1']['overall'][k])/metrics['B0']['overall'][k] for k in ('minADE','minFDE')}
    verdict='helps' if all(g>=.02 for g in gains.values()) else 'no_help' if all(g<.005 for g in gains.values()) else 'unclear'
    assert verdict==main['verdict']==result['verdict']=='no_help'
    equal(gains['minADE'],main['relative_gain_minADE'],'primary/ADE');equal(gains['minFDE'],main['relative_gain_minFDE'],'primary/FDE')
    td=read(run_dir('T1')/'TRAIN_DONE.json');bc=Path(read(run_dir('B0')/'result.json')['source'])
    assert td['contract']['train_index_sha']==__import__('hashlib').sha256(np.asarray(index,np.int64).tobytes()).hexdigest()
    assert td['train']['updates']==28800 and td['train']['parameters']==295474
    old=read(bc.parent/'metrics.json')
    assert td['train']['order_hash']==old['train']['order_hash']
    assert td['contract']['train_index_sha']==old['contract']['train_index_sha']
    assert sha(bc)==read(run_dir('B0')/'result.json')['checkpoint_sha256']
    b=data['B0'];t=data['T1'];groups={}
    for name,mask in [('overall',np.ones(364,bool)),('stable',test['group']==0),('transition',test['group']==1),('change',test['group']==2)]:
        lost=int((b['coverage'][mask]&~t['coverage'][mask]).sum());rescued=int((~b['coverage'][mask]&t['coverage'][mask]).sum())
        groups[name]=dict(n=int(mask.sum()),rescued=rescued,lost=lost,net=rescued-lost,
            ADE_relative_worsening_pct=100*(metrics['T1'][name]['minADE']/metrics['B0'][name]['minADE']-1),
            FDE_relative_worsening_pct=100*(metrics['T1'][name]['minFDE']/metrics['B0'][name]['minFDE']-1),
            ADE_delta_contribution_to_overall=float((t['minADE'][mask]-b['minADE'][mask]).sum()/364),
            FDE_delta_contribution_to_overall=float((t['minFDE'][mask]-b['minFDE'][mask]).sum()/364))
        saved=main['groups'][name]
        assert saved['n']==int(mask.sum()) and saved['rescued']==rescued and saved['lost']==lost
    speed=lambda d:np.linalg.norm(np.diff(d['past'][:,0],axis=1),axis=-1).mean(1)/.4
    speed_test=speed(test);speed_train=speed(train)[index]
    edges=np.quantile(speed_test,[0,1/3,2/3,1]);bins=np.searchsorted(edges[1:-1],speed_test,side='right')
    speed_rows=[]
    for i in range(3):
        mask=bins==i
        row=dict(bin=i,lo_mps=float(edges[i]),hi_mps=float(edges[i+1]),n=int(mask.sum()),
            B0_ADE=float(b['minADE'][mask].mean()),T1_ADE=float(t['minADE'][mask].mean()),
            B0_FDE=float(b['minFDE'][mask].mean()),T1_FDE=float(t['minFDE'][mask].mean()),
            B0_coverage=float(b['coverage'][mask].mean()),T1_coverage=float(t['coverage'][mask].mean()))
        speed_rows.append(row)
        saved=main['speed']['by_test_speed_tertile'][i]
        assert saved['n']==row['n']
        equal(row['B0_ADE'],saved['B0_minADE'],'speed/B0_ADE');equal(row['T1_ADE'],saved['T1_minADE'],'speed/T1_ADE')
        equal(row['B0_coverage'],saved['B0_cov'],'speed/B0_cov');equal(row['T1_coverage'],saved['T1_cov'],'speed/T1_cov')
    for name,values in [('train_speed',speed_train),('test_speed',speed_test)]:
        expected=dict(n=len(values),median=float(np.median(values)),p90=float(np.quantile(values,.9)))
        equal(expected,main['speed'][name],name)
    assert sum(x['n'] for x in speed_rows)==364
    partition={}
    for name,mask in [('both_good',b['coverage']&t['coverage']),('rescued',~b['coverage']&t['coverage']),('lost',b['coverage']&~t['coverage']),('both_bad',~b['coverage']&~t['coverage'])]:
        partition[name]=dict(n=int(mask.sum()),ADE_delta_sum=float((t['minADE'][mask]-b['minADE'][mask]).sum()),FDE_delta_sum=float((t['minFDE'][mask]-b['minFDE'][mask]).sum()))
    quantiles={arm:{k:{str(q):float(np.quantile(data[arm][k],q)) for q in (.5,.9,.95,.99)} for k in ('minADE','minFDE')} for arm in ('B0','T1')}
    report=dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(),verification='PASS',source_files_verified=456,final_output_files_verified=len(seal['files']),
        original_prediction_populations_recomputed=2,original_result_sha256=sha(ROOT/'ETH_T1_RESULT.json'),
        script_sha256=sha(__file__),metrics=metrics,primary_relative_improvement=gains,verdict=verdict,
        groups=groups,speed_rows=speed_rows,observed_speed_summary=main['speed'],coverage_partition=partition,error_quantiles=quantiles,
        matched_training_index_and_batch_order=True,session_seconds=result['session_wall_seconds'],
        experiment_two_decision=dict(run_now=False,status='KEEP_SHELVED',reason='T1 failed ETH primary errors; the current combo tests interaction on reused C0, not the ETH failure or a new decoder. Prioritize zero-new-training diagnosis over a new combo run.',
            single_seed_limit=True,not_a_proof_that_all_combinations_fail=True),
        scope='Existing ETH evaluation outputs only; no new training, model inference or bootstrap. Descriptive speed shift is not a proven cause.')
    output=ROOT/'analysis_inputs';output.mkdir(exist_ok=True)
    np.savez_compressed(output/'observed_speed_diagnostics.npz',ids=test['ids'],group=test['group'],test_speed_mps=speed_test,train_speed_mps=speed_train,test_speed_bin=bins,test_speed_bin_edges=edges)
    report['derived_speed_npz_sha256']=sha(output/'observed_speed_diagnostics.npz')
    (ROOT/'ANALYSIS_20261116.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(verification='PASS',source_files=456,output_hashes=len(seal['files']),verdict=verdict,
        relative_improvement=gains,group_effects=groups,coverage_partition=partition,error_quantiles=quantiles,
        speed_rows=speed_rows,experiment_two=report['experiment_two_decision']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
