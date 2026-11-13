"""Full CPU saved arithmetic for all6 models/30points; no network forward."""
import time
import numpy as np
import contracts_discrete as c
from audit_core_discrete import saved
from targets_discrete import loss_numpy
from evaluate_discrete import gates
from preflight_checks import trees_equal

def main():
    started=time.perf_counter();d=c.read_prepared('tiny');sources=c.json_read(c.ROOT/'source_manifest.json')['files'];g=c.resource(c.json_read(c.ROOT/'MEMORY_ESTIMATE.json')['CPU_estimate']);g.check(force=True);c.verify(sources,g.check)
    rows=[];events=[];summaries=[];states={};orders={};preupdate_count=0;maxloss=0.;hgate=[]
    for seed in (42,43,44):
        for arm in ('G','H'):
            folder=c.ROOT/'tiny'/f'{arm}_{seed}';done=c.json_read(folder/'done.json')
            assert done['contract']==c.contract(d,arm,'tiny')
            for rel,s in done['files'].items():assert c.sha(folder/rel)==s
            state=c.trusted_checkpoint(folder/'checkpoint_256.pt',done['files']['checkpoint_256.pt'])
            assert state['updates']==256 and state['epoch']==64 and state['cursor']==0 and len(state['history'])==64 and len(state['batch_order'])==256
            assert sum(v.numel() for v in state['model'].values())==6800
            for v in state['optimizer']['state'].values():assert int(v['step'])==256
            if seed not in states:states[seed]=state['initial'];orders[seed]=state['batch_order']
            else:assert trees_equal(states[seed],state['initial']) and orders[seed]==state['batch_order']
            trace=c.load_npz(folder/'preupdate_trace.npz');assert np.array_equal(trace['batch_indices'],state['batch_order'])
            assert trace['score'].shape==(256,32,20,48)
            for sc,ids,l in zip(trace['score'],trace['batch_indices'],trace['loss']):
                ref=loss_numpy(sc,d['value'][ids],d['valid'][ids])[0];err=abs(ref-l);maxloss=max(maxloss,err);assert err<=2e-12;preupdate_count+=1
            for point in (0,64,128,192,256):
                g.check();pointdir=folder/f'point_{point:03d}';review=saved(pointdir,d,arm,'tiny');z=c.load_npz(pointdir/'output.npz');mt=c.json_read(pointdir/'METRICS.json')
                rows.extend([dict(arm=arm,seed=seed,point=point,**r) for r in review['groups']])
                native=(d['nj']<=1).any(1)
                for i in range(128):
                    events.append(dict(arm=arm,seed=seed,point=point,sample_id=int(d['x']['sample_id'][i]),track=str(d['x']['track_key'][i]),group=int(d['x']['group'][i]),
                        native_good=bool(native[i]),good=bool(z['covered'][i]),rescue=bool(z['covered'][i]&~native[i]),lost=bool(~z['covered'][i]&native[i]),
                        choice=int(z['choice'][i]),slot=int(z['slot'][i]),action=int(z['action'][i]),selected_score=float(z['selected_score'][i]),selected_value=float(z['selected_value'][i]),
                        regret=float(z['regret'][i]),best_value=float(z['best_value'][i]),minADE=float(z['minADE'][i]),minFDE=float(z['minFDE'][i]),minJ=float(z['minJ'][i]),
                        distinct=int(z['distinct'][i]),effective_good=int(z['effective_good'][i])))
                if point==256:
                    gg=gates(review['groups'],mt['native_groups']);r=review['groups'][0]
                    summ=dict(arm=arm,seed=seed,good=r['good'],rescue=r['rescue'],lost=r['lost'],noop_fraction=mt['noop_fraction'],selected_positive=mt['selected_positive'],selected_negative=mt['selected_negative'],selected_zero=mt['selected_zero'],
                        regret=mt['regret'],balanced_MSE=mt['balanced_MSE'],unweighted_legal_MSE=mt['unweighted_legal_MSE'],groups=gg)
                    summaries.append(summ)
                    if arm=='H':
                        required={42:9,43:9,44:10}[seed];checks=dict(zero_lost=r['lost']==0,rescue_min=r['rescue']>=required,
                            nonidentity_rescue=bool(np.all(z['choice'][z['covered']&~native]>0)),protected_error=all(x.get('errors',False) for x in gg),all_motion_junction=all(x.get('motion',False) for x in gg))
                        hgate.append(dict(seed=seed,required_rescue=required,checks=checks,pass_all=all(checks.values())))
            print('REVIEW',arm,seed,'PASS',summ['good'],summ['rescue'],summ['lost'],flush=True)
    c.csv(c.ROOT/'ALL_GROUPS.csv',rows);c.csv(c.ROOT/'WINDOW_EVENTS.csv',events);c.csv(c.ROOT/'TINY_OVERVIEW.csv',[{k:v for k,v in r.items() if k!='groups'} for r in summaries])
    passed=all(r['pass_all'] for r in hgate);status='TINY_ACTION_UTILITY_PASS_ALLOW_F_OWN_ADMISSION' if passed else 'TINY_ACTION_UTILITY_STOP'
    c.write(c.ROOT/'TINY_DECISION.json',dict(status=status,H_gates=hgate,all_seed_results=summaries,primary_method='H_fixed',G_cannot_replace_failed_H=True,
        formal_models=0,formal_family5='NOT_RUN' if not passed else 'PENDING_OWN_F_IMPLEMENTATION_PREFLIGHT_BUDGET',old90_STOP_unchanged=True,no_auto_variants=True))
    c.verify(sources,g.check);g.check(force=True)
    reviewdir=c.ROOT/'completion_review';reviewdir.mkdir(exist_ok=False)
    c.write(reviewdir/'REVIEW.json',dict(status='PASS_ALL6_MODELS_30_SAVEPOINTS_SAVED_ARITHMETIC',models=6,points=30,window_outputs=30*128,candidate_outputs=30*128*20,
        preupdate_losses=preupdate_count,preupdate_double_max_error=maxloss,all_scores_choices_values_regret_masks_fixed19_quality_clusters_group_pooled_junction=True,
        paired_initial_and_batch_order_EXACT=True,all_model_optimizer_RNG_contracts=True,source_unchanged=True,
        saved_arithmetic_not_all_model_gradient_replay=True,same_assistant_not_external=True,seconds=time.perf_counter()-started))
    c.write(c.ROOT/'RESOURCE_REVIEW.json',dict(estimate=g.estimate,observations=g.observations))
    c.write(c.ROOT/'RESEARCH_REVIEW_DA2_DA3.json',dict(DA2=dict(verdict='PASS_WITH_LIMITATIONS',issue='960-way max amplifies prediction errors; balanced scores not calibrated'),
        DA3=dict(verdict='PASS',issue='guard not pooled guarantee; G already contains F16 history; repeated tiny not held generalization'),not_external_review=True))
    c.write(c.ROOT/'RESULT_ANALYSIS.json',dict(status=status,all_seed_results=summaries,H_gates=hgate,formal_models=0,
        limitations=['tiny128/97 correlated tracks','F16 historical exposure/repeated exploration','single-edit/new targets differ from oldRS','scores not probabilities','6800 excludes frozen generator/mask cost'],
        formal_family5='NOT_RUN' if not passed else 'PENDING_OWN_ADMISSION',baseline_promoted=False))
    lines=['# 有限端点动作集合效用：tiny完整结果','',f'结论：`{status}`。','',
        '| arm | seed | 覆盖/128 | 救回 | 丢失 | NOOP | 有害效用选择 |','|---|---|---|---|---|---|---|']
    for r in summaries:lines.append(f"|{r['arm']}|{r['seed']}|{r['good']}|{r['rescue']}|{r['lost']}|{r['noop_fraction']:.3f}|{r['selected_negative']}|")
    lines+=['','Native69/128；O0与新效用oracle只作GT见证。H为预先指定主方法，不能用较好G替换失败H。',
        '全部逐组误差/运动、30固定保存点及逐窗rescue-lost/选择分数与真实效用见ALL_GROUPS.csv/WINDOW_EVENTS.csv。未调阈值、seed、目标系数或epoch。',
        '本轮移除的是连续回归/量化表示，不回填旧C约束结论；与旧RS同时改变表示、监督和单槽政策，非纯单因素比较。',
        'G/H仅显式历史16维不同；G原生候选已有F16历史信息。此tiny配对结果不等于正式H-G迁移检验。',
        '正式H1-H5及hold/inner结果尚未运行，原90%STOP/major/部署保护门与baseline均保持。',
        '同助手ARS DA1/2/3和独立NumPy算术不是外审，不是全部模型梯度二次回放。',
        '若T失败，本有限系列结束；不因失败追加训练或切换最好G进入F。全部文件/失败/模型/日志保留。','']
    with (c.ROOT/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    c.write(c.ROOT/'EXECUTION_ACCOUNTING.json',dict(tiny_models=6,tiny_updates=1536,tiny_epochs=384,preflight_optimizer_updates=c.json_read(c.ROOT/'PREFLIGHT.json')['fixture_updates'],formal_models=0,source_preflight_failures_separately_disclosed=True))
    files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts and 'logs' not in p.parts
        and p.name not in ('WORKER_START.json','WORKER_STATE.json','WORKER_EXIT.json','runtime_control.json','startup_check.json','halfway_check.json','completion_check.json','task.xml')}
    c.write(c.ROOT/'FINAL_PRESERVATION.json',dict(status='TINY_SCIENCE_AND_SOURCE_SEAL',files=files,logs_runtime_sealed_after_worker_exit=True))
    for rel,s in files.items():assert c.sha(c.ROOT/rel)==s
    c.write(c.ROOT/'FINAL_SEAL_CHECK.json',dict(status='PASS',files=len(files),manifest=c.sha(c.ROOT/'FINAL_PRESERVATION.json')))
    print('TINY FINAL',status,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import datetime,traceback
        c.write(c.ROOT/('FAILURE_REVIEW_'+datetime.datetime.now().strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc()))
        raise
