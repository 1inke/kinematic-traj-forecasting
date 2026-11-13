"""CPU saved arithmetic/CI/source review, not another model or gradient replay."""
from common import *
from evaluate_saved import production,independent
from statistics_fixed import primary,gates
import csv

def main():
    setup();out=HERE/'completion_review';out.mkdir(exist_ok=False);g=Guard(read(HERE/'MEMORY_ESTIMATE.json')['CPU_E']);start=time.perf_counter();data_all=data();mf=read(HERE/'source_manifest.json')
    metrics={};samples={};audits=[];rows=[];native={};checkpoints=[];rootfiles={}
    for split in ('FIT','inner'):
        p=HERE/f'native/{split}';r=read(p/'done.json');verify({str(p/k):s for k,s in r['files'].items()},g.check);z=load_npz(p/'predictions.npz');er=load_npz(p/'sample_metrics.npz');nm=read(p/'metrics.json')
        assert np.array_equal(z['paths'],data_all[split]['native']);audit=independent(z['paths'],data_all[split],nm,er,g);audits.append(dict(arm='native',split=split,**audit));native[split]=nm
    for s in SEEDS:
        i1=torch.load(HERE/f'initial/Q1_{s}.pt',weights_only=True);i3=torch.load(HERE/f'initial/Q3_{s}.pt',weights_only=True)
        assert all(torch.equal(v,i3[k]) for k,v in i1.items())
        orders=load_npz(HERE/f'orders/{s}.npz')['orders'];history_pair=[]
        for arm in ('Q1','Q3'):
            run=HERE/f'runs/{arm}_{s}';receipt=done(run/'done.json',contract(arm,s));hist=read(run/'history.json');assert len(hist)==8 and sum(x['steps'] for x in hist)==592
            batchids=[[digest(data_all['FIT']['ids'][order[i:i+256]]) for i in range(0,18909,256)] for order in orders]
            assert [[b['IDs'] for b in h['batches']] for h in hist]==batchids
            for h in hist:
                counts=np.asarray(h['winner_counts']);assert np.all(counts.sum(1)==18909)
                assert np.all(np.asarray(h['tied_windows'])<=18909)
            history_pair.append(batchids)
            cp=trusted(run/'checkpoint.pt',receipt['files'][(run/'checkpoint.pt').relative_to(HERE).as_posix()],contract(arm,s));assert cp['epoch']==8 and cp['cursor']==0
            export=torch.load(run/'export_main_group0.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(v,export[k]) for k,v in cp['model'].items() if not k.startswith('aux.'))
            assert sum(v.numel() for k,v in cp['model'].items() if k not in ('mean','std'))=={'Q1':82008,'Q3':82648}[arm]
            checkpoints.append(dict(arm=arm,seed=s,epochs=8,updates=592,checkpoint=sha(run/'checkpoint.pt'),export=sha(run/'export_main_group0.pt'),all_state_contract_initial_order_history_exact=True))
            for split,pop in data_all.items():
                stage('Q_CPU_COMPLETION_REVIEW',arm=arm,seed=s,split=split);p=run/split;z=load_npz(p/'predictions.npz');e=load_npz(p/'sample_metrics.npz');m=read(p/'metrics.json')
                for k,key in [('ids','sample_id'),('gt','gt'),('native','native'),('past','past'),('groups','groups'),('track','track')]:assert np.array_equal(pop[k],z[key]),key
                assert z['all_group_paths'].shape==(len(pop['ids']),{'Q1':1,'Q3':3}[arm],20,12,2)
                assert np.array_equal(z['paths'],z['all_group_paths'][:,0]);audit=independent(z['paths'],pop,m,e,g);audits.append(dict(arm=arm,seed=s,split=split,**audit))
                metrics[arm,s,split]=m;samples[arm,s,split]=dict(coverage=e['coverage'],groups=z['groups'],track=z['track'],ids=z['sample_id'])
                for group,r in m.items():rows.append(dict(arm=arm,seed=s,split=split,group=group,N=r['n'],native_good=r['native_good'],good=r['good'],rescue=r['rescue'],lost=r['lost'],ADE=r['ADE'],FDE=r['FDE'],J=r['J'],distinct=r['distinct'],effective_good=r['effective_good'],**r['motion']))
                # Recompute every auxiliary group's complete same-candidate coverage;
                # auxiliary cluster/unionK60 are not main deployment outcomes.
                aux=read(p/'auxiliary_diagnostic.json');uc=[]
                for group in range(z['all_group_paths'].shape[1]):
                    dd=np.linalg.norm(z['all_group_paths'][:,group].astype(float)-pop['gt'][:,None],axis=-1);good=(dd.mean(-1)<=.2)&(dd[:,:,-1]<=.4);uc.append(good.any(1))
                    assert int(good.any(1).sum())==aux['groups'][str(group)]['overall']['good']
                assert int(np.stack(uc).any(0).sum())==aux['union_GK_not_deployment']['good']
                del z,e;g.check(force=True)
            del cp
        assert history_pair[0]==history_pair[1]
    a=np.stack([samples['Q3',s,'inner']['coverage'] for s in SEEDS]);b=np.stack([samples['Q1',s,'inner']['coverage'] for s in SEEDS]);pop=data_all['inner']
    stage('Q_PRIMARY_20000_ALL_DRAW_INDEPENDENT_REVIEW')
    stat=primary(a,b,pop['groups'],pop['track'],HERE/'BOOTSTRAP_DRAWS.npz');write(HERE/'PRIMARY_COMPARISON.json',stat);decision=gates(stat,metrics,samples,native);write(HERE/'DECISION.json',decision)
    with (HERE/'ALL_GROUPS.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'ARITHMETIC_AUDIT.json',dict(status='PASS',population_jobs=14,scientific_model_jobs=12,native_jobs=2,all_reports=audits,checkpoints=checkpoints,all_bootstrap_draws_maxerror=stat['independent_all_draw_maxerror'],all_saved_not_full_model_gradient_replay=True))
    stage('Q_COMPLETION_FINAL_ALL_SOURCE_SHA');verify(mf['files'],g.check);g.check(force=True)
    write(out/'RESOURCE.json',dict(status='PASS',policy='warning_first_remaining_v3',E=g.estimate,observations=g.observations))
    write(out/'PRESERVATION.json',dict(status='PASS_ALL_ORIGINAL_SOURCE_DIGESTS',count=len(mf['files']),digest=mf['digest'],old_outputs_unchanged=True))
    write(HERE/'RESULT.json',dict(status='SUCCESS_Q_COMPLETE_REVIEW_REQUIRED',science=decision['status'],scientific_models=6,scientific_epochs=48,scientific_updates=3552,
        population_jobs=14,main_inference_K=20,train_parameters=[82008,82648],export_parameters=82008,arithmetic='PASS',source='PASS',resource='PASS',M=decision['M'],E=decision['E'],baseline_promoted=False))
    report(stat,decision,rows)
    for p in list(HERE.glob('*.json'))+list(HERE.glob('*.csv'))+list(HERE.glob('*.npz'))+list(out.glob('*.json')):
        if p.name in ('progress.json','runtime_control.json','WORKER_STATE.json','WORKER_START.json','WORKER_EXIT.json','source_manifest.json'):continue
        rootfiles[str(p.relative_to(HERE))]=sha(p)
    write(out/'ROOT_OUTPUTS_SHA.json',dict(status='PASS_EXPLICIT_ROOT_STATISTICS_RESOURCE_RESULT_AND_AUDITS',files=rootfiles))
    write(out/'REVIEW.json',dict(status='PASS_COMPLETE_SAVED_CPU_REVIEW',seconds=time.perf_counter()-start,source_count=len(mf['files']),population_jobs=14,checkpoint_models=6,all_draws=20000,
        root_output_seal=sha(out/'ROOT_OUTPUTS_SHA.json'),review='same assistant independent algorithms not external peer review'))
    print(decision['status'],flush=True)

def report(stat,decision,rows):
    body=['# 分组查询 Q 阶段实际结果','',f"执行与保存审核完成；科学门：{decision['status']}。",'',
        'Q1单组与Q3三组：seed42/43/44，各fresh8epoch；FIT18909、inner1856；部署固定组0 K20。新参数82008/82648，部署82008。原端点线性hinge不变。', '',
        '## 主比较：inner change，同一候选ADE≤0.2且FDE≤0.4','',f"每seed覆盖差(pp)：{[100*x for x in stat['per_seed_delta']]}；平均{100*stat['mean_delta']:.4f}pp。",f"固定family3校正98.3333%区间：{stat['paired_seed_track']['corrected_family3']}；固定seed敏感性：{stat['fixedseed_track_sensitivity']['corrected_family3']}。区间单位为比例。",'',
        '|seed|split|group|N|Native|Q1|Q3|Q1 lost native|Q3 lost native|','|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for s in SEEDS:
        for split in ('FIT','inner'):
            for group in ('overall','stable','transition','change'):
                a=next(r for r in rows if r['arm']=='Q1' and r['seed']==s and r['split']==split and r['group']==group);b=next(r for r in rows if r['arm']=='Q3' and r['seed']==s and r['split']==split and r['group']==group)
                body.append(f"|{s}|{split}|{group}|{a['N']}|{a['native_good']}|{a['good']}|{b['good']}|{a['lost']}|{b['lost']}|")
    body+=['','所有连续误差、原成功保留、完整GTfree聚簇、有效good和五pooled运动学/接续见ALL_GROUPS.csv及每人口metrics/sample_metrics；相对门逐条在DECISION.json。辅助组与K60 union只作不等预算诊断。', '',
        f"Q3 Native严格安全：{decision['Q3_native_safety']}。M：{decision['M']}；E：{decision['E']}。机制门不是Native安全或重大突破门；不升级baseline。",'',
        '## 同助手反方核查（不是外部评审）','',
        'DA2：Q3增加640训练参数及三组decoder计算。即便有收益，仍混合了训练计算/优化几何影响，不能证明纯监督数量的因果中介。绑定相同query夹具只排除loss乘3/跨60取min的实现伪效应。完整预算和吞吐见BENCHMARK.json。', '',
        'DA3：主组无提升不能推论观测不含信息；辅助组或并集提升也不能冒充K20部署增益。须保留FIT与inner分歧、每seed反例、lost和运动退化；不挑最好seed/组或缩统计家族。', '',
        '限制：旧F16 h3/v2已见全部FIT及inner；目标track不交不保证上下文不交；inner反复用于开发，三seed相关窗口bootstrap仍是探索，不是独立wholemodel泛化、心理意图理解或SOTA。保存数据重算不是全部模型/梯度再次回放。AI助手完成实施和分角色核查，非外审。', '',
        '原major双5%、原成功lost≤1%、protected误差1%、五motion/junction1%与相关性审核均不放宽；旧baseline/historycycle11/长期未完成目标保持。无额外组数/宽度/epoch/损失扫描。']
    with (HERE/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(body)+'\n')
    write(HERE/'RESULT_ANALYSIS.json',dict(primary=stat,decision=decision,all_rows=rows,inlineDA2='computation and parameter confounding disclosed',inlineDA3='all seeds/groups/error/retention/motion retained; no generalization overclaim',external_review=False))
if __name__=='__main__':main()
