"""Finite final interpretation of already sealed tiny outputs, no new experiment."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import contracts_discrete as c

def main():
    old=c.json_read(c.ROOT/'FINAL_PRESERVATION_V2.json')
    for rel,s in old['files'].items():assert c.sha(c.ROOT/rel)==s
    d=c.read_prepared('tiny');qN=d['nj'].min(1);rows=[];cases=[];neutral=[]
    base=c.json_read(c.ROOT/'tiny/H_42/point_000/METRICS.json')['native_groups']
    for seed in (42,43,44):
        for arm in ('G','H'):
            f=c.ROOT/'tiny'/f'{arm}_{seed}/point_256';z=c.load_npz(f/'output.npz');mt=c.json_read(f/'METRICS.json')
            for r,n in zip(mt['groups'],base):
                rows.append(dict(arm=arm,seed=seed,group=r['group'],N=r['N'],good=r['good'],rescue=r['rescue'],lost=r['lost'],
                    ADE_harm_percent=100*(r['minADE']/n['minADE']-1),FDE_harm_percent=100*(r['minFDE']/n['minFDE']-1),
                    errors_pass=all(r[k]<=n[k]*(1 if r['group']=='change' else 1.01) for k in ('minADE','minFDE')),
                    motion_pass=all(r[k]<=n[k]*1.01 for k in c.reference.NAMES)))
            iszero=z['selected_value']==0;unchanged=z['minJ']==qN;clip=iszero&~unchanged&(qN>2)&(z['minJ']>2)
            neutral.append(dict(arm=arm,seed=seed,NOOP=int((z['choice']==0).sum()),selected_positive=int((z['selected_value']>0).sum()),selected_negative=int((z['selected_value']<0).sum()),
                selected_zero=int(iszero.sum()),zero_and_set_minJ_unchanged=int((iszero&unchanged).sum()),zero_hides_change_both_clipped_above2=int(clip.sum())))
            olderr=np.linalg.norm(d['x']['native']-d['gt'][:,None],axis=-1);oldF=olderr[:,:,-1].min(1)
            for i in range(128):
                if int(d['x']['group'][i])==1 and z['minFDE'][i]>oldF[i]:
                    cases.append(dict(arm=arm,seed=seed,sample_id=int(d['x']['sample_id'][i]),track=str(d['x']['track_key'][i]),choice=int(z['choice'][i]),slot=int(z['slot'][i]),action=int(z['action'][i]),
                        old_minFDE=float(oldF[i]),new_minFDE=float(z['minFDE'][i]),delta_minFDE=float(z['minFDE'][i]-oldF[i]),native_minJ=float(qN[i]),new_minJ=float(z['minJ'][i]),
                        native_good=bool(qN[i]<=1),new_good=bool(z['covered'][i]),predicted_score=float(z['selected_score'][i]),true_utility=float(z['selected_value'][i])))
    c.csv(c.ROOT/'completion_extension/FINAL_GROUP_FAILURE_BREAKDOWN.csv',rows)
    c.csv(c.ROOT/'completion_extension/ALL_TRANSITION_FDE_HARM_CASES.csv',cases)
    c.csv(c.ROOT/'completion_extension/SELECTED_UTILITY_DECOMPOSITION.csv',neutral)
    c.write(c.ROOT/'completion_extension/INTERPRETATION.json',dict(status='PASS_SEALED_SAVED_INTERPRETATION',all_group_rows=rows,selected_utility=neutral,
        all_transition_FDE_harm_cases=cases,new_models=0,new_updates=0,held_GT_evaluation=False,source_sha=c.sha(__file__),prior_seal_sha=c.sha(c.ROOT/'FINAL_PRESERVATION_V2.json')))
    prior=(c.ROOT/'RESULT_ANALYSIS_FINAL.md').read_text(encoding='utf-8')
    text='\n\n## 结果的具体含义（封存输出补充解读）\n\n'
    text+='1. 新表示确实产生了执行编辑：六模型NOOP均0，不再是F-RS的执行identity停滞；但与旧RS不是单因素因果比较。H相对配对G每seed多救回2窗，lost差为−2/0/−2，有限动作regret也各seed较低。这是训练内线索，不是正式H−G迁移显著性结果。\n\n'
    text+='2. 最关键的change组：Native9/23，六模型均仍9/23，rescue/lost均0；新效用oracle在该组可到12/23（3窗机会）。真实学习收益主要来自stable和少量transition，不能把overall净增当成change已改善。\n\n'
    text+='3. H三seed transition minFDE相对Native恶化约3.233%/3.233%/4.766%，超过1%保护门；不是运动学失败。H42/H44零lost也不能掩盖连续误差伤害。全部逐窗反例见ALL_TRANSITION_FDE_HARM_CASES.csv。\n\n'
    text+='4. H实际选择正/负/零效用分别13/2/113、8/5/115、10/3/115；大部分高分编辑真实效用为0，少数为负。960选1与非校准分数风险得到具体表现，但没有通过额外干预证明它是唯一根因。minJ效用为0也不保证独立minADE/minFDE不恶化。\n\n'
    text+='5. 因此本轮不能说“完全没学到”，也不能说集合保护已解决。按预注册T门停止F，停止这一固定小端点分支，不做后验阈值/系数/epoch修补。下一大方向需另立明确设计。\n'
    with (c.ROOT/'FINAL_REPORT.md').open('x',encoding='utf-8') as f:f.write(prior+text)
    c.write(c.ROOT/'FINAL_REPORT.json',dict(c.json_read(c.ROOT/'RESULT_ANALYSIS_FINAL.json'),interpretation=c.json_read(c.ROOT/'completion_extension/INTERPRETATION.json'),canonical_report='FINAL_REPORT.md'))
    c.write(c.ROOT/'ANALYZED.json',dict(status='ANALYZED_TINY_ACTION_UTILITY_STOP',formal='NOT_TRIGGERED_T_GATE_FAILED',canonical='FINAL_REPORT.md',new_models=6,tiny_updates=1536,formal_updates=0,no_next_automatic_training=True))
    c.verify(c.json_read(c.ROOT/'source_manifest.json')['files'])
    files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts and p.name not in ('runtime_control.json','startup_check.json','halfway_check.json','completion_check.json')}
    c.write(c.ROOT/'FINAL_PRESERVATION_V3.json',dict(status='COMPLETE_FINAL_REPORT_AND_INTERPRETATION_SEAL',files=files,all_previous_seals_retained=True))
    for rel,s in files.items():assert c.sha(c.ROOT/rel)==s
    c.write(c.ROOT/'FINAL_SEAL_CHECK_V3.json',dict(status='PASS',files=len(files),manifest=c.sha(c.ROOT/'FINAL_PRESERVATION_V3.json')))
    print('FINAL INTERPRETATION SEALED',len(files),flush=True)
if __name__=='__main__':main()
