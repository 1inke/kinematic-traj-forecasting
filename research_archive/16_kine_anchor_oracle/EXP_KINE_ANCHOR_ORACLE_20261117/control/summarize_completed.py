from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
cfg=read(ROOT/'CONFIG.json');out=Path(cfg['output_dir']);r=read(out/'RESULT.json');verification=read(ROOT/'INDEPENDENT_VERIFICATION.json');assert verification['status']=='PASS'
for p,h in read(ROOT/'VERIFICATION_INPUT_CONTRACT.json')['files'].items():assert sha(p)==h,p
def metric(fold,split,arm,group,key):return float(np.mean([v[group][key] for v in r['folds'][fold]['splits'][split][arm]]))
nums=r['decision']['numbers'];dec=nums['decel_stop_minFDE'];ethdec={a:metric('eth','test',a,'decel_stop','minFDE') for a in ('P','P_AUG','S','K')}
lines=['# 机动模板零训练检验：结果解读','',
f"**实验完成，判定 NOT_SUPPORTED。** 5折、4主臂、6个敏感性配置、3个聚类seed，共150次聚类，正式计算{r['seconds']:.1f}秒（约{r['seconds']/60:.2f}分钟）。没有神经网络训练或推理，未使用GPU。10项自检、负对照和独立真实数据几何复算均通过。",'',
'这个结果表示本次固定20模板的机动表示没有通过预设综合门槛，不表示所有运动学解码思路都无效。没有修改阈值、改选敏感性配置或启动后续训练。','',
'P是位置模板，P_AUG是增强后的同类模板，S是按当前有效速度归一化的位置模板，K是三段速度比与方向模板。所有模板都只在各折95% fit上聚类，val/test候选只依赖模板和观测速度/朝向，未来仅用于误差计算及评价标签。','',
'| 固定检查 | 结论 | 依据 |','|---|---|---|',
f"| C1 ETH收益 | 通过 | K相对P，ADE改善{100*nums['eth_test_g_ADE']:.2f}%，FDE改善{100*nums['eth_test_g_FDE']:.2f}% |",
f"| C2 不差于增强 | 未通过 | 五折等权FDE，K {nums['fivefold_test_minFDE_main']:.4f}，P_AUG {nums['fivefold_test_minFDE_P_AUG']:.4f} |",
'| C3 不伤其他折 | 未通过 | HOTEL/UNIV/ZARA01/ZARA02的FDE分别恶化2.94%/20.43%/28.53%/30.42% |',
f"| C4 分布内不变差 | 未通过 | 五折val等权ADE恶化{-100*nums['fivefold_val_g_ADE']:.2f}%，FDE恶化{-100*nums['fivefold_val_g_FDE']:.2f}% |",
f"| C5 减速/停止 | 未通过 | 合并FDE：K {dec['main']:.4f}，P {dec['P']:.4f}，P_AUG {dec['P_AUG']:.4f} |",
'| C6 ETH三seed一致 | 通过 | 三个聚类seed的FDE均优于对应P |','',
'SUPPORT和WEAK都要求C3、C4通过，本轮两项均失败，所以NOT_SUPPORTED不是因某个边界值轻微未达标。','',
'ETH单折的改善不能概括成机动参数化普遍更好：S的ETH FDE为0.8004，比K的0.8258还低约3.17%；P_AUG为0.7788，也好于K。K相对S在合并减速/停止组又有改善，归因结果为MIXED，不能把全部收益归给机动参数化或单纯速度归一化。','',
f"本来希望补强的减速/停止也没有稳定获益：五折合并K相对P恶化{100*(dec['main']/dec['P']-1):.2f}%；ETH该组97窗，P {ethdec['P']:.4f}、K {ethdec['K']:.4f}、P_AUG {ethdec['P_AUG']:.4f}m，K相对P恶化{100*(ethdec['K']/ethdec['P']-1):.2f}%。ETH模板列表确有逐段减速的模板，但“存在减速模板”不足以保证它适合这些窗口。",'',
'按预定路线，本次不支持直接为这一K表示投入训练。后续仍可优先研究候选间优化、评分与校准；若探索更好的速度尺度处理，应另行固定设计再检验。本轮敏感性中S的V0=.5表现较好，只保留为后续假设，不事后替换V0=.3的主结果。','',
'已训练B0/AUG和用真值选比例的P仅为参考。后者在17个固定尺度中以ADE选比例，并保留该比例的minFDE；不是连续尺度的严格理论上界，也不能部署。三次seed是聚类初始化seed，不是三次网络训练重复。','',
'原REPORT.md/RESULT.json、TEMPLATES.npz和10份PER_WINDOW文件保持原字节；analysis_inputs另存10份主臂、敏感性、尺度参考及已有模型的逐seed误差（不含原始坐标）。p90和可达率必须先按seed计算再平均，不能直接从平均误差反推。DERIVED_INPUTS.json说明格式。','',
'独立复核从原始数据和150份保存模板重建全部主臂/敏感性候选与尺度参考，核对10个val/test人群的全部机动、速度、变化组指标，重算C1–C6、归因和最终结论；未重新聚类。脚本和回执在同级EXP_KINE_ANCHOR_ORACLE_CONTROL_20261117目录。正式任务由Windows服务托管，exit0，父进程链与GPT无关。','',
'仍属探索：测试数据已被反复查看，模板只有速度和朝向，未用网络上下文；K采用角度的欧氏聚类，跨±π及低速方向处理等表示局限仍在。这个比较检验的是当前具体设计，不能给整个研究方向下普遍否定结论。','']
(out/'RESULT_INTERPRETATION_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
write(out/'EXECUTION_PROVENANCE.json',dict(control_dir=str(ROOT),original_design_dir=str(ROOT/'design_original'),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),formal_output_seal_sha256=sha(ROOT/'DONE.json'),verification_receipt_sha256=sha(ROOT/'INDEPENDENT_VERIFICATION.json'),verification_script_sha256=sha(ROOT/'verify_completed.py'),result_status='COMPLETE_VERIFIED',scientific_verdict='NOT_SUPPORTED',new_training_runs=0,new_inference_runs=0,GPU_used=False))
all_output={str(p):sha(p) for p in out.rglob('*') if p.is_file()}
control_files=[ROOT/'DONE.json',ROOT/'SOURCE_MANIFEST.json',ROOT/'PREFLIGHT.json',ROOT/'INDEPENDENT_VERIFICATION.json',ROOT/'VERIFICATION_INPUT_CONTRACT.json',ROOT/'verify_completed.py',ROOT/'summarize_completed.py',ROOT/'WORKER_EXIT.json',ROOT/'PROCESS_ANCESTRY.json']
write(ROOT/'COMPLETION_VERIFIED.json',dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(),status='COMPLETE_VERIFIED',scientific_verdict='NOT_SUPPORTED',fits_completed=150,source_files=47,formal_original_outputs=14,self_tests=10,independent_verification='PASS',files={**all_output,**{str(p):sha(p) for p in control_files}}))
write(ROOT/'CURRENT.json',dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(),phase='COMPLETE_VERIFIED',verdict='NOT_SUPPORTED',fits_completed=150))
print(json.dumps(dict(status='COMPLETE_VERIFIED',verdict='NOT_SUPPORTED',original_outputs=14,all_output_files=len(all_output),output_dir=str(out)),ensure_ascii=False))
