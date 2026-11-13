"""Saved independent review only; no model, no training, no DPP sampling."""
import argparse,time
from common import *
from scoring import *
from independent import score_loop,spectral_ec_loop,quality_loop
import numpy as np
from threadpoolctl import threadpool_limits

def main(execute=False):
    if not authorize(execute,'review'):return
    source=read(RUN/'source_manifest.json');verify(source['files'])
    computed=read(RUN/'STAGE_A_COMPUTED.json');verify(computed['outputs'])
    c=read(RUN/'CALIBRATION.json');reports={};rg=guard_module();guard=rg.Guard(2*1024**3)
    with threadpool_limits(limits=1):
        for name in MODELS:
            for split in ('fit','dev'):
                p,m=checked_population(name,split);folder=RUN/'populations'/name/split
                receipt=read(folder/'done.json');verify(receipt['files'])
                with np.load(folder/'diagnostics.npz',allow_pickle=False) as f:s={k:f[k] for k in f.files}
                if not np.array_equal(s['ids'],p['ids']) or not np.array_equal(s['groups'],p['groups']):raise ValueError('SAVED_ID_GROUP_MISMATCH')
                maxq=maxec=maxraw=maxmarginal=maxrawscaled=0.
                for start in range(0,len(p['ids']),64):
                    sl=slice(start,min(start+64,len(p['ids'])));y=p['paths'][sl].astype(np.float64);past=p['past'][sl].astype(np.float64)
                    q,valid,raw=score_loop(past,y,c,return_details=True)
                    maxq=max(maxq,float(np.max(abs(q-s['qkin'][sl]))))
                    if not np.array_equal(valid,s['kin_valid'][sl]):raise ValueError('SAVED_VALIDITY')
                    for key,value in raw.items():
                        diff=abs(value-s[key][sl]);maxraw=max(maxraw,float(np.max(diff)))
                        maxrawscaled=max(maxrawscaled,float(np.max(diff/(2e-12+2e-12*abs(value)))))
                    for key,quality in [('EC_uniform',np.ones_like(q)),('EC_kin',q)]:
                        ec,marginal=spectral_ec_loop(y,quality,c['k'],c['epsilon_q'],return_marginal=True)
                        maxec=max(maxec,float(np.max(abs(ec-s[key][sl]))))
                        marginal_key='marginal_uniform' if key=='EC_uniform' else 'marginal_kin'
                        maxmarginal=max(maxmarginal,float(np.max(abs(marginal-s[marginal_key][sl]))))
                    if not np.isnan(s['qscene'][sl]).all() or not np.isnan(s['qjoint'][sl]).all() or s['scene_known_fraction'][sl].any():raise ValueError('MAPLESS_JOINT_QUALITY_FABRICATION')
                    guard.check()
                if maxq>2e-12 or maxec>2e-12 or maxrawscaled>1 or maxmarginal>2e-12:raise ValueError('SAVED_SECOND_IMPLEMENTATION_FAILURE')
                reports[name+'/'+split]=dict(N=len(p['ids']),all_candidate_score_error=maxq,raw_penalty_logq_error=maxraw,raw_scaled_arithmetic_error=maxrawscaled,all_EC_second_loop_error=maxec,all_marginal_spectral_error=maxmarginal,scene_unknown_preserved=True)
                del p,m,s
    verify(source['files']);verify(computed['outputs'])
    write(RUN/'INDEPENDENT_REVIEW.json',dict(status='PASS_ALL_SAVED_KIN_SCORES_AND_ALL_DPP_EC_INDEPENDENT_LOOPS',populations=reports,no_full_model_or_gradient_replay=True,source_unchanged=True))
    result=dict(status='COMPLETE_STAGE_A_KIN_DIAGNOSIS_SCENE_AND_STAGE_B_UNRESOLVED',at=now(),new_trained_models=0,
        scene='BLOCKED_MISSING_ADMITTED_METRIC_MAP',paper_B0_radius='NOT_COMPUTABLE_NO_DSF_GAUSSIAN_LATENTS',stageB='NOT_STARTED_NO_AUTOMATIC_TRAINING',
        no_candidate_change=True,no_oracle_accuracy_gain_possible=True,same_assistant_not_external=True,
        ARS_DA2='Scoring tracks against thresholds derived from same FIT distribution is not independent validation. A lower q for high own-penalty tracks is a definition, not evidence of physical safety or generalization.',
        ARS_DA3='Fixed paths mean no coverage/minADE/FDE improvement by scoring alone. Valid-only spread and expected cardinality do not establish learned coverage. Missing scene branch is unresolved, not negative. Related windows and same seed42 sources are not independent trials.',
        original_baseline_historycycle11_goal_unchanged=True)
    write(RUN/'RESULT.json',result)
    text=['# DPP双重可行性：阶段A离线诊断','',
        '运动学质量与DPP算术完成，场景分支及DSF训练未执行。新训练模型0；没有选择/删除/替换候选。原K20覆盖及minADE/FDE保持不变。',
        '完整逐模型/fit-dev/原C0组与GT转向组分母见ALL_GROUPS.csv。qscene与qjoint均未知，不能称双重可行性已验证。',
        '',result['ARS_DA2'],result['ARS_DA3'],'',
        '经验阈值来自FIT GT99.5%和IQR，不是普适生理边界。评分下限1e-6仅DPP质量数值选项，raw/logq与下溢数均保存；没有dev扫参。',
        '无合格场景图、原DSF接口和真实latent，因此完整B0-B4训练比较未实施。下一阶段须实际补齐这些合同，不能用h3嵌入伪装Gaussian latent或换模型凑实验。','',
        '全部文件、失败、旧结果和PAUSED记录保留；同助手ARS反证非外审。']
    with (RUN/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    roots=[RUN/n for n in ('RESULT.json','RESULT_ANALYSIS.md','INDEPENDENT_REVIEW.json','STAGE_A_COMPUTED.json','RESOURCE.json','ALL_GROUPS.csv','ARITHMETIC_AUDIT.json','PREFLIGHT_REAL.json','CALIBRATION.json')]
    write(RUN/'FINAL_SEAL.json',dict(status='PASS_STAGE_A_SAVED_DIAGNOSIS_ONLY',files=seal(roots),sources=source['files']))
    print('DPP_OFFLINE_STAGE_A_REVIEW_COMPLETE')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)
