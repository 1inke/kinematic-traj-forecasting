"""Bounded phases for experiment one; reuse B0 and train only T1 seed42."""
import argparse
import sys
import gc
from types import SimpleNamespace
from eth_exec_common import *


def audit_outputs(directory,rf,cfg):
    import numpy as np
    from fivefold_metrics import per_window,aggregate
    test=rf.load_split(cfg['data_dir'],'eth','test')
    with np.load(directory/'test_predictions.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
    with np.load(directory/'test_window_metrics.npz',allow_pickle=False) as z:w={k:z[k] for k in z.files}
    assert p['paths'].shape==(364,20,12,2) and p['paths'].dtype==np.float32
    assert p['logits'].shape==(364,20) and np.isfinite(p['paths']).all() and np.isfinite(p['logits']).all()
    for key in ('ids','group'):assert np.array_equal(p[key],test[key]) and np.array_equal(w[key],test[key])
    calc=per_window(p['paths'],p['logits'],test['future'],test['past'][:,0])
    for key,values in calc.items():assert np.allclose(values,w[key],rtol=1e-12,atol=1e-12),key
    metrics=aggregate(calc,test['group']);past=test['past'][:,0].astype(np.float64)
    v1=(p['paths'][:,:,0].astype(np.float64)-past[:,None,-1])/.4
    metrics['overall']['junction_accel_allK20']=float((np.linalg.norm(v1-((past[:,-1]-past[:,-2])/.4)[:,None],axis=-1)/.4).mean())
    expected=read(directory/'result.json')['test']
    for group,block in metrics.items():
        assert block.keys()==expected[group].keys()
        for key,value in block.items():
            if value is None:assert expected[group][key] is None
            else:assert np.isclose(value,expected[group][key],rtol=1e-12,atol=1e-12),(group,key)
    replace_json(directory/'EVALUATION_AUDIT.json',dict(at=now(),status='PASS',n=364,K=20,source='saved float32 predictions recomputed against exact ETH targets'))


def reference():
    ec,original=imports();tc,rf,guard=ec.load_training()
    config=ec.read(ec.PARALLEL/'PARALLEL_CONFIG.json')
    out=run_dir('B0')
    if (out/'EVALUATION_SEAL.json').exists():validate_seal(out,'EVALUATION_SEAL.json');return
    import torch
    train=rf.load_split(config['data_dir'],'eth','train');idx,_=rf.scene_tail_split(train['scene'])
    old,checkpoint,state=original.b0_checkpoint(ec,config,rf,idx)
    verify(read(old/'DONE.json')['files'])
    test=rf.load_split(config['data_dir'],'eth','test')
    trainer=tc.Trainer(train,idx,batch=64,epochs=64,device='cuda:0',seed=42,arm='B0')
    trainer.load(state,state['contract']);guard.check(force=True)
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'result.json').exists():
        metrics=original.evaluate_model(ec,tc,guard,trainer.ema.module,test,out)
        replace_json(out/'result.json',dict(at=now(),arm='B0',source=str(checkpoint),checkpoint_sha256=sha(checkpoint),fold='eth',epochs=64,seed=42,test=metrics))
    else:
        j=read(out/'result.json')
        assert j['checkpoint_sha256']==sha(checkpoint)
    audit_outputs(out,rf,config);write_seal(out,'EVALUATION_SEAL.json')


def train():
    ec,original=imports();out=run_dir('T1')
    if (out/'TRAIN_SEAL.json').exists():validate_seal(out,'TRAIN_SEAL.json');return
    original.train(SimpleNamespace(explore_dir=str(EXPLORE),out=str(ROOT)))
    tc,rf,guard=ec.load_training();cfg=ec.read(ec.PARALLEL/'PARALLEL_CONFIG.json')
    data=rf.load_split(cfg['data_dir'],'eth','train');idx,_=rf.scene_tail_split(data['scene'])
    expected=original.contract(ec,rf,cfg,idx);done=read(out/'TRAIN_DONE.json')
    assert done['contract']==expected and done['train']['updates']==28800
    ck=sorted(out.glob('ckpt_epoch*.json'))[-1];receipt=read(ck)
    assert receipt['epoch']==64 and receipt['updates']==28800 and sha(receipt['file'])==receipt['sha256']
    write_seal(out,'TRAIN_SEAL.json')


def evaluate():
    validate_seal(run_dir('B0'),'EVALUATION_SEAL.json');validate_seal(run_dir('T1'),'TRAIN_SEAL.json')
    ec,original=imports();out=run_dir('T1')
    if (out/'EVALUATION_SEAL.json').exists():validate_seal(out,'EVALUATION_SEAL.json');return
    original.evaluate(SimpleNamespace(explore_dir=str(EXPLORE),out=str(ROOT)))
    tc,rf,guard=ec.load_training();cfg=ec.read(ec.PARALLEL/'PARALLEL_CONFIG.json')
    audit_outputs(out,rf,cfg)
    write_seal(out,'EVALUATION_SEAL.json')


def analyze():
    import numpy as np
    validate_seal(run_dir('B0'),'EVALUATION_SEAL.json');validate_seal(run_dir('T1'),'EVALUATION_SEAL.json')
    ec,original=imports();tc,rf,guard=ec.load_training()
    sys.path.insert(0,str(ROOT/'code'))
    import analyze_eth_t1 as analysis
    # Descriptive speed bins must be disjoint and use the actual 95% training subset.
    def speed_breakdown(fivefold_code,data_dir,wb,wt):
        test=rf.load_split(data_dir,'eth','test');data=rf.load_split(data_dir,'eth','train');idx,_=rf.scene_tail_split(data['scene'])
        assert np.array_equal(test['ids'],wb['ids'])
        speed=lambda d:np.linalg.norm(np.diff(d['past'][:,0],axis=1),axis=-1).mean(1)/.4
        ste=speed(test);straining=speed(data)[idx];edges=np.quantile(ste,[0,1/3,2/3,1])
        bins=np.searchsorted(edges[1:-1],ste,side='right');rows=[]
        for i in range(3):
            mask=bins==i
            rows.append(dict(speed_mps=f'{edges[i]:.2f}-{edges[i+1]:.2f}',n=int(mask.sum()),B0_minADE=float(wb['minADE'][mask].mean()),T1_minADE=float(wt['minADE'][mask].mean()),B0_cov=float(wb['coverage'][mask].mean()),T1_cov=float(wt['coverage'][mask].mean())))
        assert sum(x['n'] for x in rows)==len(ste)
        q=lambda x:dict(n=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)))
        return dict(train_speed=q(straining),test_speed=q(ste),by_test_speed_tertile=rows)
    analysis.speed_breakdown=speed_breakdown
    cfg=read(ec.PARALLEL/'PARALLEL_CONFIG.json')
    sys.argv=['analyze_eth_t1.py','--out',str(ROOT),'--fivefold-code',str(ec.FIVE/'code'),'--data-dir',cfg['data_dir']]
    analysis.main()
    decision=read(ROOT/'ETH_T1_RESULT.json')
    replace_json(ROOT/'INTERPRETATION.json',dict(at=now(),scope='Single seed exploratory ETH evaluation, not untouched confirmatory test',
        verdict_seed42=decision['verdict'],automatic_seed43=False,
        if_unclear='Report seed42 as inconclusive; retain proposal to add paired B0/T1 seed43, without silently expanding this initial stage.',
        baseline_upgraded=False,COMBO_started=False,speed_bin_amendment='Disjoint tertiles; train-speed reference uses the actual scene-tail 95% subset. Primary decision unchanged.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['reference','train','evaluate','analyze']);args=parser.parse_args()
    verify_source()
    if args.phase in ('reference','train'):
        directory=run_dir('B0' if args.phase=='reference' else 'T1');directory.mkdir(parents=True,exist_ok=True)
        binding=directory/'RUN_SOURCE.json'
        if binding.exists():assert read(binding)['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
        else:replace_json(binding,dict(at=now(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json')))
    status(ROOT/'standalone/slots'/f'{args.phase}.json',dict(status='RUNNING',phase=args.phase))
    globals()[args.phase]();verify_source()
    status(ROOT/'standalone/slots'/f'{args.phase}.json',dict(status='DONE',phase=args.phase))
