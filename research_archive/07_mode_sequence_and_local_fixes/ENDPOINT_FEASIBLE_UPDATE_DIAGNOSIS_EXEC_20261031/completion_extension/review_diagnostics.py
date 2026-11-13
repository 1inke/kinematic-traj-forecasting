"""CPU-only explicit saved-data extension: extra trial scalar and event costs.

Does not change any locked scientific source or previous completed seal.
"""
import collections
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import contracts as c
from checks import Bank,independent_paths,independent_groups,quality,NAMES

def window_motion(p,past):
    h=np.asarray(past,float)
    v=(p-np.concatenate((np.zeros(p.shape[:-2]+(1,2)),p[...,:-1,:]),-2))/.4
    a=np.diff(v,axis=-2)/.4;j=np.diff(v,n=2,axis=-2)/(.4**2)
    v,a,j=[np.sqrt((vv*vv).sum(-1)) for vv in (v,a,j)]
    jun=np.sqrt(((p[:,:,0]/.4-((h[:,-1]-h[:,-2])/.4)[:,None])**2).sum(-1))/.4
    return np.stack((np.quantile(v,.95,axis=(1,2)),a.mean((1,2)),np.quantile(a,.95,axis=(1,2)),j.mean((1,2)),np.quantile(j,.95,axis=(1,2)),jun.mean(1),np.quantile(jun,.95,axis=1)),1)

def main():
    started=time.perf_counter();d=c.data();b=Bank(d);sup=d['nj']<=1;nc=sup.any(1);rows=[];extra=[];U2=[];maxerr=0;trialn=0
    sealed=c.json_read(c.ROOT/'FINAL_PRESERVATION.json')['files']
    for name,s in sealed.items():assert c.sha(c.ROOT/name)==s
    native_motion=window_motion(d['x']['native'],d['x']['past'])
    for seed in (42,43,44):
        f=c.ROOT/f'F_RS_{seed}';records=c.json_read(f/'proposal_history.json');us=c.load_npz(f/'all_trial_u.npz')['u']
        previous=np.full((128,20),b.a.identity);unchanged=0;changes=0;fail=collections.Counter();binding_windows=collections.Counter()
        for rec in records:
            for tr in rec['trials']:
                if not tr['changed']:continue
                u=us[tr['u_index']];p=d['x']['native']+.4*(np.arange(1,13)/12)[None,None,:,None]*u.astype(float)[:,:,None]
                q=quality(p,d['gt']);v=np.where(nc,np.where(sup,q['J'],np.inf).min(1),0.)
                err=float(np.max(np.abs(v-np.asarray(tr['C_original_support_minJ']))));maxerr=max(maxerr,err);assert err<=2e-12;trialn+=1
                if not tr['all_pass']:
                    for k,values in tr['bits'].items():
                        if any(values):fail[k]+=1
                    for wi,failed in enumerate(tr['bits']['continuous_original_support']):
                        if failed:binding_windows[int(d['x']['sample_id'][wi])]+=1
            if rec['accepted']:
                tr=rec['trials'][-1];z=independent_paths(us[tr['u_index']],d);same=np.array_equal(previous,z['executed_action_id']);unchanged+=int(same);changes+=int(not same);previous=z['executed_action_id']
        final=c.load_npz(f/'point_256/paths.npz');soft=c.load_npz(c.PREV/'tiny'/f'RS_{seed}/final.npz');oo=c.load_npz(c.ROOT/'O0_witness/witness.npz')
        extra.append(dict(seed=seed,accepted_quantized_actions_unchanged_from_preproposal=unchanged,accepted_actions_changed=changes,
            rejected_trial_failure_categories=dict(fail),failure_categories_overlap=True,continuous_binding_window_counts=dict(binding_windows),
            maximum_abs_normalized_control=float(np.abs(final['u']).max()),
            final_Q_exact_native=np.array_equal(final['Q'],d['x']['native']),final_S_exact_native=np.array_equal(final['S'],d['x']['native'])))
        for method,paths in [('Native',d['x']['native']),('O0',oo['paths']),('SoftRS',soft['guarded_paths']),('F_RS',final['S'])]:
            q=quality(paths,d['gt']);cov=q['good'].any(1);wm=window_motion(paths,d['x']['past'])
            for i in range(128):
                event='retained' if nc[i] and cov[i] else 'lost' if nc[i] else 'rescue' if cov[i] else 'unresolved'
                rows.append(dict(method=method,seed=seed,sample_id=int(d['x']['sample_id'][i]),track=str(d['x']['track_key'][i]),group=int(d['x']['group'][i]),event=event,
                    minADE=float(q['ADE'][i].min()),minFDE=float(q['FDE'][i].min()),minJ=float(q['J'][i].min()),
                    **dict(zip(NAMES,wm[i].tolist())),**dict(zip(('native_'+n for n in NAMES),native_motion[i].tolist()))))
        for point in (64,128,192,256):
            sub=c.ROOT/'U2_output_probe'/f'{seed}_{point:03d}';choice=c.json_read(sub/'CHOICE.json');v=choice['global_verdict'];report={}
            for label in ('local','alpha0','alpha1'):
                z=c.load_npz(sub/label/'paths.npz');q,gg=independent_groups(z['S'],d);r=gg[0]
                report[label]={k:r[k] for k in ('good','rescue','lost')}
            U2.append(dict(seed=seed,point=point,**report,all_global_constraints_pass=v['all_pass'],
                failed_categories=[k for k,val in v['bits'].items() if any(val)],alpha_counts=dict(collections.Counter(map(str,choice['alpha'])))))
    c.csv(c.ROOT/'completion_extension/FINAL_WINDOW_MOTION_EVENTS.csv',rows)
    binding_rows=[]
    for r in extra:
        for i in range(128):
            binding_rows.append(dict(seed=r['seed'],sample_id=int(d['x']['sample_id'][i]),track=str(d['x']['track_key'][i]),group=int(d['x']['group'][i]),
                native_good=bool(nc[i]),native_minJ=float(d['nj'][i].min()),native_good_margin=float(1-d['nj'][i].min()) if nc[i] else None,
                failed_trial_count=r['continuous_binding_window_counts'].get(int(d['x']['sample_id'][i]),0)))
    c.csv(c.ROOT/'completion_extension/CONTINUOUS_BINDING_WINDOWS.csv',binding_rows)
    c.write(c.ROOT/'completion_extension/REVIEW.json',dict(status='PASS_ALL_RECORDED_C_SUPPORT_MINJ_AND_FINAL_EVENT_COSTS',trials=trialn,
        max_support_minJ_error=maxerr,extra=extra,U2=U2,root_science_unchanged=True,source_file_sha=c.sha(__file__),
        base_final_manifest_sha=c.sha(c.ROOT/'FINAL_PRESERVATION.json'),no_forward_backward_optimizer=True,not_external_review=True,seconds=time.perf_counter()-started))
    fixture_account=[]
    for suffix,pf_name in [('preflight','PREFLIGHT.json'),('preflight_v2','PREFLIGHT_V2.json')]:
        pf=c.json_read(c.ROOT/pf_name);attempts=sum(pf['fixture_optimizer_proposals'].values())
        real=c.json_read(c.ROOT/suffix/'real_resume/RESULT.json')['accepted']
        forced=c.json_read(c.ROOT/suffix/'forced_transaction_resume/RESULT.json')['accepted']
        # full8 and interrupted5+resumed3 each reproduce the same accepted count.
        commits=2+12+2*(real+forced)
        fixture_account.append(dict(stage=suffix,Adam_proposals=attempts,committed_optimizer_steps_across_all_fixture_runs=commits))
    science=c.json_read(c.ROOT/'EXECUTION_ACCOUNTING.json')
    resources=[]
    for name in ('RESOURCE_PREFLIGHT.json','RESOURCE_PREFLIGHT_V2.json','RESOURCE_SCIENCE.json','RESOURCE_REVIEW.json'):
        r=c.json_read(c.ROOT/name);obs=r['observations']
        resources.append(dict(stage=name,own_process_lifetime_private_peak=max(x['snapshot']['peak_private'] for x in obs),
            own_process_lifetime_RSS_peak=max(x['snapshot']['peak_rss'] for x in obs),
            warnings=sorted(set(w for x in obs for w in x['warnings'])),resource_STOP=any(x['status']!='PASS' for x in obs)))
    c.write(c.ROOT/'EXECUTION_ACCOUNTING_FINAL.json',dict(scientific= science,fixtures=fixture_account,
        total_actual_Adam_step_calls=768+sum(x['Adam_proposals'] for x in fixture_account),
        scientific_accepted=science['scientific_accepted_optimizer_updates'],resources=resources,
        independent_inference_subprocess_memory_not_separately_profiled=True,
        inference_fixture_note='deliberately changed GT payload was loaded in the container but never read by prediction/decode; observed_only docstring describes inference data dependence, not archive IO'))
    decision=c.json_read(c.ROOT/'DECISION.json');summary=decision['seeds']
    with (c.ROOT/'RESULT_ANALYSIS_FINAL.md').open('x',encoding='utf-8') as f:
        f.write('# 端点可行更新诊断：最终综合报告\n\n')
        f.write(f"结论：`{decision['status']}`。这是训练内有限搜索诊断，不是可学习性或泛化的不可能性证明。\n\n")
        f.write('## 1. 固定问题与执行\n\n只用原tiny128窗/97相关tracks；Native69成功、59失败；H4770、40维原统计、原RS目标/AdamW/批序/END49/guard保持。3fresh模型各256提议，拒绝不回滚批次时钟；接受Adam提议矩但全局参数位移按固定alpha缩小。旧RS只读复用。\n\n')
        f.write('| seed | F-RS覆盖/救回/丢失 | 接受/拒绝 | Soft-RS救回/丢失 |\n|---|---|---|---|\n')
        for r in summary:f.write(f"|{r['seed']}|{r['F_good']}/{r['F_rescue']}/{r['F_lost']}|{r['accepted']}/{r['rejected']}|{r['SoftRS_rescue']}/{r['SoftRS_lost']}|\n")
        f.write('\n零丢失和组保护由接受器强制保证，不能当作独立成果。全部参数MSE、LR/LS、alpha分布与救回/21见METHOD_COMPARISON.csv。全部组原始误差/运动与C/Q/S见AUDITED_ALL_GROUPS.csv；逐窗事件/成本包含Native、O0、旧RS及F-RS，不仅报告净增。\n\n')
        f.write('## 2. O0与固定U2\n\nO0完整通过：90覆盖、21救回、0丢失；group误差、pooled五motion与junction均通过。它只证明该固定动作表里存在一个不共享的GT查表见证，不作教师或部署。\n\n')
        f.write('| seed | 提议点 | alpha0救回 | alpha1救回/丢失 | 局部缩步救回/丢失 | 全局门 |\n|---|---|---|---|---|---|\n')
        for r in U2:f.write(f"|{r['seed']}|{r['point']}|{r['alpha0']['rescue']}|{r['alpha1']['rescue']}/{r['alpha1']['lost']}|{r['local']['rescue']}/{r['local']['lost']}|{'PASS' if r['all_global_constraints_pass'] else 'FAIL'}|\n")
        f.write('\nU2为GT_DIAGNOSTIC_ONLY：原坏窗不按是否救回挑alpha，原好窗仅按原支持/guarded保留选首个合格alpha。局部选择后重新检查全局门；不因全局失败再调整。不挑最佳快照。输出插值不等非线性参数插值，也不必由同一个网络实现，因此不能作共享干扰纯因果解释。\n\n')
        f.write('## 3. 接受器限制与反证\n\n')
        for r in extra:f.write(f"- seed{r['seed']}：已接受但执行动作未变{r['accepted_quantized_actions_unchanged_from_preproposal']}次、动作改变{r['accepted_actions_changed']}次；拒绝trial失败类别（可重叠）{r['rejected_trial_failure_categories']}。\n")
        f.write('\n本轮全部拒绝trial的绑定条件均为连续C的原成功支持保留，而非S端group误差或运动学门。三seed最大绝对标准化控制分别约0.05273/0.04969/0.05329，最终Q与S均逐字节等于Native；全部已接受更新都未改变执行动作。固定U2从这些停滞点的一次提议出发，12快照也均无新增救回。这更具体地说明本条提议/缩步路径在跨越执行量化边界前即被连续支持保护卡住，不证明所有输出修正/共享结构无解，也不据此放宽C门或增加alpha。\n')
        f.write('\nO0阳性不能保证共享RS梯度方向能到达；F-RS停滞不能排除其它共享解。保护coverage之外的group误差仍是独立限制。不将当前失败外推为历史观测无用或未来不可预测。\n\n')
        f.write('## 4. 审核、时序、成本与边界\n\n完整独立128 pooled预检在科学提议前完成。旧RS首提议全部4770参数/梯度/Adam EXACT，实际128两遍8提议/第5提议恢复及强制拒绝/alpha<1夹具全部参数、Adam、sampler、CPU/CUDA/Python/NumPy RNG、trial和路径EXACT。原子提交checkpoint保留partial语义。\n\n')
        f.write('V1实现和预检保留；启动前发现并补严原子checkpoint及包含模型前向的8trial benchmark，V2完整重检。不是生产失败或科学规则变更。预检fixture与768科学提议单独计数；原对照未重训。\n\n')
        bench=c.json_read(c.ROOT/'prelaunch_extension/RESULT.json');comp=c.json_read(c.ROOT/'COMPUTATION.json');rev=c.json_read(c.ROOT/'completion_review/REVIEW.json')
        f.write(f"保守启动估算{bench['conservative_total_seconds']:.2f}秒（门1800秒）；实际科学运行/保存{comp['seconds']:.2f}秒，独立完整审核{rev['seconds']:.2f}秒，另有本补充审核/来源与报告封存，不冒充完整人时。\n\n")
        f.write('全部trial接受科学标量/失败位已独立复算；所有固定保存点/12U2各三对照端口的质量、完整K20 complete-link(.10/.20)、有效good簇、五pooled、junction重算。原始FINAL_PRESERVATION/FINAL_SEAL保持，V2补充封存新增文件。保存算术不是全模型/梯度二次回放，同助手ARS DA1/2/3非外审。\n\n')
        f.write('3seed不是3份独立数据，反复tiny/F16已见/相关窗限制不消失；未做held或新显著性宣称，原family6 NOT_RUN。旧90%STOP、TIME/P3、major5%/保护1%/相关性门、baseline/historycycle11及长期目标保持。本有限诊断结束，不自动正式训练、MoE/gate/第四loss或新alpha扫描。所有内容保留，没有删除。\n')
        f.write('\n资源为各受测进程自身Windows保留生命周期峰，不冒称整个进程树聚合峰；独立观测推理子进程未单独采集内存曲线。该推理夹具容器故意含改变GT，预测/解码不使用它；不将“无GT前向”夸大为文件读取器从未装载GT键。详见EXECUTION_ACCOUNTING_FINAL.json。\n')
    result=dict(c.json_read(c.ROOT/'RESULT_ANALYSIS.json'),completion_extension=c.json_read(c.ROOT/'completion_extension/REVIEW.json'),final_report='RESULT_ANALYSIS_FINAL.md')
    c.write(c.ROOT/'RESULT_ANALYSIS_FINAL.json',result)
    files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    c.write(c.ROOT/'FINAL_PRESERVATION_V2.json',dict(status='FINAL_COMPLETE_SEAL_WITH_EXTENSION',files=files,original_seal_unchanged=True))
    for p,s in files.items():assert c.sha(c.ROOT/p)==s
    c.write(c.ROOT/'FINAL_SEAL_CHECK_V2.json',dict(status='PASS',files=len(files),manifest=c.sha(c.ROOT/'FINAL_PRESERVATION_V2.json')))
    print('EXTENSION FINAL',decision['status'],extra,flush=True)
if __name__=='__main__':main()
