"""Full saved arithmetic and finite-series conclusion; no model forward/training."""
import collections
import time
import numpy as np
import contracts as c
from checks import Bank,compare,independent_paths,independent_verdict,independent_groups,full_metrics,quality,NAMES
from evaluate_saved import read_port
from objectives import numpy_loss_and_grad

def main():
    t=time.perf_counter();d=c.data();bank=Bank(d);g=c.resource();g.check(force=True)
    sources=c.json_read(c.ROOT/'source_manifest.json')['files'];c.old.verify(sources,g.check)
    allgroups=[];events=[];strata=[];summaries=[];trialcount=0;loss_errors=[]
    nc=(d['nj']<=1).any(1);ns=d['nj']<=1;sn=ns.sum(1)
    oo=c.load_npz(c.ROOT/'O0_witness/witness.npz');oz,orr=full_metrics(oo['paths'],d)
    compare(oz,{k:oo[k] for k in oz});assert int((oz['covered']&~nc).sum())==21
    def inspect_port(folder,method,seed,point):
        z=read_port(folder,d);out=independent_paths(z['u'],d)
        for k in ('u','C','Q','S','raw_action_id','executed_action_id','failed'):compare(z[k],out[k])
        if method=='F_RS':assert independent_verdict(out['C'],out['S'],d)['all_pass']
        for port in ('C','Q','S'):
            metrics,rows=full_metrics(z[port],d)
            for k,v in metrics.items():compare(z[port+'_'+k],v)
            allgroups.extend([dict(method=method,seed=seed,point=point,port=port,**r) for r in rows])
            for i in range(128):
                events.append(dict(method=method,seed=seed,point=point,port=port,sample_id=int(d['x']['sample_id'][i]),track=str(d['x']['track_key'][i]),group=int(d['x']['group'][i]),
                    native_good=bool(nc[i]),good=bool(metrics['covered'][i]),rescue=bool(~nc[i]&metrics['covered'][i]),lost=bool(nc[i]&~metrics['covered'][i]),
                    minJ=float(metrics['minJ'][i]),distinct=int(metrics['distinct'][i]),effective_good=int(metrics['effective_good'][i])))
        return z
    for seed in (42,43,44):
        folder=c.ROOT/f'F_RS_{seed}';done=c.json_read(folder/'done.json');c.verify_contract(done['contract'],d)
        for rel,s in done['files'].items():assert c.sha(folder/rel)==s
        state=c.trusted_checkpoint(folder/'checkpoint_256.pt',done['files']['checkpoint_256.pt'])
        assert state['transaction']=='COMMITTED_ONLY' and state['updates']==256 and state['accepted']+state['rejected']==256
        assert len(state['batch_order'])==256 and len(state['history'])==64
        for vv in state['optimizer']['state'].values():assert int(vv['step'])==state['accepted']
        trialus=c.load_npz(folder/'all_trial_u.npz')['u'];records=c.json_read(folder/'proposal_history.json');trace=c.load_npz(folder/'preupdate_trace.npz')
        used=[]
        for p,rec in enumerate(records):
            g.check();ids=trace['batch_indices'][p];u=trace['u'][p]
            _,_,parts=numpy_loss_and_grad(u,d['target'][ids],d['x']['native'][ids],d['gt'][ids],d['best'][ids],'RS')
            actual=((u-d['target'][ids])**2).mean((1,2)).mean();err=abs(float(actual)-rec['components']['mse']);assert err<=2e-7;loss_errors.append(err)
            for k in ('retention','rescue'):assert abs(parts[k]-rec['components'][k])<=2e-12
            for j,tr in enumerate(rec['trials']):
                if not tr['changed']:assert tr['reason']=='ZERO_DISPLACEMENT';continue
                index=tr['u_index'];used.append(index);u=trialus[index]
                out=independent_paths(u,d);vv=independent_verdict(out['C'],out['S'],d)
                compare(tr,vv);trialcount+=1
                if vv['all_pass']:assert j==len(rec['trials'])-1 and rec['accepted'] and rec['alpha']==tr['alpha']
            assert rec['accepted']==bool(rec['alpha'])
        assert used==list(range(len(trialus)))
        for point in (0,64,128,192,256):z=inspect_port(folder/f'point_{point:03d}','F_RS',seed,point)
        final=z;soft=c.load_npz(c.PREV/'tiny'/f'RS_{seed}/final.npz')
        rescue=int((final['S_covered']&~nc).sum());lost=int((~final['S_covered']&nc).sum());assert lost==0
        parts=numpy_loss_and_grad(final['u'],d['target'],d['x']['native'],d['gt'],d['best'],'RS')[2]
        alpha=collections.Counter(str(r['alpha']) for r in records if r['accepted'])
        summaries.append(dict(seed=seed,F_good=int(final['S_covered'].sum()),F_rescue=rescue,F_lost=lost,SoftRS_good=int(soft['S_covered'].sum()),SoftRS_rescue=int((soft['S_covered']&~nc).sum()),SoftRS_lost=int((~soft['S_covered']&nc).sum()),
            eliminated_old_losses=int((nc&~soft['S_covered']&final['S_covered']).sum()),paired_rescue_delta=rescue-int((soft['S_covered']&~nc).sum()),
            accepted=state['accepted'],rejected=state['rejected'],accepted_alpha_counts=dict(alpha),
            zero_displacement_trials=sum(not tr['changed'] for r in records for tr in r['trials']),
            rescue_fraction_of21=rescue/21,nonidentity_fraction=float((final['executed_action_id']!=bank.a.identity).mean()),
            action_unchanged_from_native=bool((final['executed_action_id']==bank.a.identity).all()),final_continuous_proxies=parts))
        typ=np.where(ns,np.where(sn[:,None]==1,0,1),np.where(sn[:,None]>0,2,3))
        for k,name in enumerate(('unique_support','multiple_support','badslot_in_good_window','native_bad_window')):
            m=typ==k;strata.append(dict(seed=seed,stratum=name,candidates=int(m.sum()),windows=int(m.any(1).sum()),
                parameter_MSE=float(((final['u']-d['target'])**2).mean(-1)[m].mean()),support_slots_lost=int((m&ns&~final['S_good']).sum())))
        for point in (64,128,192,256):
            sub=c.ROOT/'U2_output_probe'/f'{seed}_{point:03d}';inp=c.load_npz(sub/'inputs.npz');choice=c.json_read(sub/'CHOICE.json')
            pending=np.ones(128,bool);chosen=np.zeros(128);uout=np.empty_like(inp['u_current'])
            for a in c.ALPHAS+(0.,):
                u=inp['u_current']+a*(inp['u_full_proposal']-inp['u_current']);out=independent_paths(u,d)
                qc=quality(out['C'],d['gt']);qs=quality(out['S'],d['gt'])
                local=~nc|((ns&(qc['J']<=1)).any(1)&qs['good'].any(1))
                local &= np.isfinite(u).all((1,2))&np.isfinite(out['S']).all((1,2,3))
                take=pending&local;chosen[take]=a;uout[take]=u[take];pending[take]=False
            assert not pending.any();compare(chosen,choice['alpha'])
            for label in ('local','alpha0','alpha1'):
                z=inspect_port(sub/label,'U2_'+label,seed,point)
                if label=='local':compare(z['u'],uout);compare(choice['global_verdict'],independent_verdict(z['C'],z['S'],d))
        print('REVIEW COMPLETE',seed,'rescue',rescue,'accepted',state['accepted'],flush=True)
    c.csv(c.ROOT/'METHOD_COMPARISON.csv',summaries);c.csv(c.ROOT/'SUPPORT_STRATA.csv',strata);c.csv(c.ROOT/'WINDOW_EVENTS.csv',events)
    c.csv(c.ROOT/'AUDITED_ALL_GROUPS.csv',allgroups)
    anyres=any(r['F_rescue']>0 for r in summaries);nontrivial=all(r['F_rescue']>=1 for r in summaries)
    promising=nontrivial and all(r['F_rescue']>=r['SoftRS_rescue'] for r in summaries)
    status='PROMISING_TINY_FEASIBLE_UPDATE' if promising else 'NONTRIVIAL_FEASIBLE_SHARED_WITNESS' if nontrivial else 'LIMITED_FEASIBLE_TRADEOFF' if anyres else 'IDENTITY_OR_STALLED'
    u2=[r for r in allgroups if r['method']=='U2_local' and r['port']=='S' and r['group']=='overall']
    c.write(c.ROOT/'DECISION.json',dict(status=status,seeds=summaries,O0=c.json_read(c.ROOT/'O0_witness/RESULT.json')['status'],U2_all_fixed_snapshots=u2,formal_models=0,old90STOP_unchanged=True,no_auto_followon=True))
    # Same-assistant inline DA, never external peer review.
    da=dict(DA2=dict(verdict='PASS_WITH_LIMITATIONS',issue='line search changes effective step and Adam timing; U2 removes parameter realizability',
        strongest_counterargument='a fixed failed search cannot rule out feasible shared models'),DA3=dict(verdict='PASS',issue='O0/U2 use GT; U1 accepts on repeatedly explored training128',
        disclosure='same assistant self-review, not external; no held results or formal significance'))
    c.write(c.ROOT/'RESEARCH_REVIEW_DA2_DA3.json',da)
    c.old.verify(sources,g.check);g.check(force=True)
    rr=c.ROOT/'completion_review';rr.mkdir(exist_ok=False)
    c.write(rr/'REVIEW.json',dict(status='PASS_ALL_SAVED_ARITHMETIC',trial_count=trialcount,all_trial_bits_and_scalars_recomputed=True,
        all15_savepoints_and36_U2_ports_full_quality_clusters_pooled=True,O0_all_groups=True,proposals=768,loss_MSE_backend_max=max(loss_errors),
        source_unchanged=True,not_all_model_or_gradient_replay=True,not_external_review=True,seconds=time.perf_counter()-t))
    c.write(c.ROOT/'RESOURCE_REVIEW.json',dict(estimate=g.estimate,observations=g.observations))
    c.write(c.ROOT/'RESULT_ANALYSIS.json',dict(status=status,seeds=summaries,U2=u2,DA=da,O0=c.json_read(c.ROOT/'O0_witness/RESULT.json'),
        limitations=['training128/97 correlated tracks','three seeds not independent data','repeated tiny exploration/F16 previously seen','U2 nonnetwork output witness','zero lost is acceptance tautology'],
        formal_models=0,major_or_baseline_promotion=False))
    lines=['# 端点可行更新诊断：完整结果','',f'结论：`{status}`。仅固定tiny128/97 tracks训练内诊断；无正式训练、无held评估、无baseline升级。','',
        '| seed | F-RS覆盖/救回/丢失 | Soft-RS救回/丢失 | 接受/拒绝 |','|---|---|---|---|']
    for r in summaries:lines.append(f"| {r['seed']} | {r['F_good']}/{r['F_rescue']}/{r['F_lost']} | {r['SoftRS_rescue']}/{r['SoftRS_lost']} | {r['accepted']}/{r['rejected']} |")
    lines+=['','Native覆盖69/128，safe-oracle可救回21窗；全部结果与逐窗事件见JSON/CSV。',
        f"O0：{c.json_read(c.ROOT/'O0_witness/RESULT.json')['status']}，覆盖90/救回21/丢失0；是GT查表，不是可部署预测。",'',
        '零丢失与组保护是接受条件强制结果，不是独立发现。实际科学问题是约束下剩余救回及接受数。',
        'U2固定12快照全部报告（JSON）；不挑最佳快照、不把输出空间见证归因纯共享干扰。',
        '全部试验trial接受标量/位由第二NumPy实现复算；所有保存点完整K20聚簇、same-slot质量、五pooled运动与junction复算。保存算术不是全部模型梯度再回放。',
        '反方自检为同助手，非外审。3seed不构成3份独立数据；反复tiny和F16历史暴露限制泛化解释。',
        '本有限系列结束，不自动全量训练、MoE/gate/第四loss或新alpha集合；原90%STOP/major/部署门保持。','']
    with (c.ROOT/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    pf=c.json_read(c.ROOT/'PREFLIGHT.json')
    accepted=sum(r['accepted'] for r in summaries)
    c.write(c.ROOT/'EXECUTION_ACCOUNTING.json',dict(scientific_proposals=768,scientific_accepted_optimizer_updates=accepted,scientific_rejected_proposals=768-accepted,
        new_models=3,preflight_fixture_proposals=pf['fixture_optimizer_proposals'],formal_models=0,
        preflight_fixed_math_has_no_optimizer_update=True,O0_U2_CPU_no_training=True))
    c.write(c.ROOT/'runtime_state.json',dict(status='ANALYZED_FINITE_DIAGNOSIS_COMPLETE',decision=status,proposals=768,accepted=accepted,no_auto_formal=True))
    files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    c.write(c.ROOT/'FINAL_PRESERVATION.json',dict(status='FINAL_SOURCE_OUTPUT_SEAL',files=files,source_manifest=c.sha(c.ROOT/'source_manifest.json')))
    for p,s in files.items():assert c.sha(c.ROOT/p)==s
    c.write(c.ROOT/'FINAL_SEAL_CHECK.json',dict(status='PASS',files=len(files),manifest=c.sha(c.ROOT/'FINAL_PRESERVATION.json')))
    print('FINAL DECISION',status,flush=True)

if __name__=='__main__':main()
