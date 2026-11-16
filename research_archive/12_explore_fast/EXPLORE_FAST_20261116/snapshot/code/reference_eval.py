"""Read-only reuse of the five chosen budget checkpoints; separate exploratory outputs."""
import gc
import numpy as np
from explore_common import *


def main():
    verify_source()
    tc,rf,guard=load_training()
    import torch
    import runtime as rt
    config=read(PARALLEL/'PARALLEL_CONFIG.json');out=Path(config['output'])
    seal=read(out/'DECISION_SEAL.json')
    assert sha(out/'BUDGET_DECISION.json')==seal['decision_sha256']
    assert read(out/'BUDGET_DECISION.json')['E_star']==128
    rows=[]
    for fold in ('eth','hotel','univ','zara01','zara02'):
        destination=ROOT/'budget_reference'/fold
        destination.mkdir(parents=True,exist_ok=True)
        run=out/'runs'/f'budget_{fold}_E128_s42'
        assert sha(run/'DONE.json')==seal['budget_done_sha256'][run.name]
        rt.validate_done(run)
        completed=destination/'result.json'
        if completed.exists():
            result=read(completed);verify(result['files']);rows.append(result);continue
        status(ROOT/'standalone/slots/reference.json',dict(status='EVALUATE_OLD_BUDGET',fold=fold))
        train=rf.load_split(config['data_dir'],fold,'train');indices,_=rf.scene_tail_split(train['scene'])
        checkpoint=sorted(run.glob('ckpt_epoch*.pt'))[-1]
        receipt=read(checkpoint.with_suffix('.json'));assert sha(checkpoint)==receipt['sha256']
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['epoch']==128 and state['contract']['fold']==fold
        assert state['contract']['train_index_sha']==rf.index_sha(indices)
        trainer=tc.Trainer(train,indices,batch=64,epochs=128,device='cuda:0',seed=42,arm='B0')
        trainer.load(state,state['contract'])
        test=rf.load_split(config['data_dir'],fold,'test')
        guard.check(force=True)
        paths,logits=tc.infer(trainer.ema.module,test,guard=guard)
        from fivefold_metrics import per_window,aggregate
        paths=paths.astype(np.float32);logits=logits.astype(np.float32)
        window=per_window(paths,logits,test['future'],test['past'][:,0])
        metrics=aggregate(window,test['group'])
        save_npz(destination/'test_predictions.npz',paths=paths,logits=logits,ids=test['ids'],group=test['group'])
        save_npz(destination/'test_window_metrics.npz',ids=test['ids'],group=test['group'],**window)
        result=dict(at=now(),fold=fold,epochs=128,seed=42,train_fraction=.95,
                    selection='FIXED_FINAL_EMA',test=metrics,checkpoint=str(checkpoint),
                    checkpoint_sha256=receipt['sha256'],scope='EXPLORATORY_SINGLE_SEED_REFERENCE',
                    files={str(p):sha(p) for p in destination.glob('*.npz')})
        replace_json(completed,result);rows.append(result)
        del trainer,state,train,test,paths,logits,window
        gc.collect();torch.cuda.empty_cache()
    m2=read(Path(config['exp156']))['rows'];m2={r['fold']:r for r in m2}
    averages={k:float(np.mean([r['test']['overall'][k] for r in rows])) for k in ('minADE','minFDE')}
    m2avg={k:float(np.mean([r[key] for r in m2.values()])) for k,key in (('minADE','min_ade_20'),('minFDE','min_fde_20'))}
    summary=dict(at=now(),status='EXPLORATORY_REFERENCE_COMPLETE',rows=rows,equal_weight=averages,
                 EXP156_single_seed_reference=m2avg,not_the_unfinished_three_seed_result=True)
    replace_json(ROOT/'BUDGET_REFERENCE.json',summary)
    lines=['# 已暂停五折实验的阶段结论','',
      '原五折三seed实验尚未完成。以下主表是复用已有95%训练、seed42、128轮预算模型得到的探索性参考；没有新增训练。',
      'EXP156的模型、训练和选checkpoint方式不同，只作量级参照。','',
      '| 折 | B0 minADE/minFDE | EXP156 M2 minADE/minFDE |','|---|---:|---:|']
    for r in rows:
        f=r['fold'];v=r['test']['overall'];m=m2[f]
        lines.append(f"| {f} | {v['minADE']:.4f}/{v['minFDE']:.4f} | {m['min_ade_20']:.4f}/{m['min_fde_20']:.4f} |")
    lines += [f"| 五折等权 | {averages['minADE']:.4f}/{averages['minFDE']:.4f} | {m2avg['minADE']:.4f}/{m2avg['minFDE']:.4f} |",'',
      '已经完成的原100%训练正式结果：','', '| ETH seed | minADE/minFDE |','|---|---:|']
    for seed in (42,43):
        d=out/'runs'/f'final_eth_E128_s{seed}';rt.validate_done(d)
        v=read(d/'metrics.json')['test']['overall']
        lines.append(f"| {seed} | {v['minADE']:.4f}/{v['minFDE']:.4f} |")
    lines += ['', '预算验证集显示64→128平均相对ADE改善2.2169%、平均FDE变化约−0.008米；这是单seed验证结果。',
      '原最终15次中仅ETH两个seed完成，其余结果未齐，不能给出计划中的五折三seed均值和标准差。',
      '未完成的ETH44和hotel42断点、全部原结果保留；本次按用户要求暂停，不自动恢复。',
      'C0开发数据与四折测试存在相同目标轨迹窗口；标准五折结果不等同全新独立验证。',
      '后续优先进行短预算探索，覆盖率变化同时检查旧好窗口丢失和运动变化；没有升级baseline。']
    (ROOT/'PARTIAL_CONCLUSION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status(ROOT/'standalone/slots/reference.json',dict(status='DONE',equal_weight=averages))
    print(json.dumps(dict(status=summary['status'],equal_weight=averages,reference=m2avg),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
