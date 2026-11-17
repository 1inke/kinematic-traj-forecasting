"""Readable findings from sealed results and the independent verification receipt."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'DECEL_DIAG.json').read_text(encoding='utf-8'))
v=json.loads((ROOT/'INDEPENDENT_VERIFICATION.json').read_text(encoding='utf-8'))
assert v['status']=='PASS'
def dec(model):return r['datasets']['ETH'][model]['cohorts']['deceleration']['all_history']
b=dec('B0_s42');a=dec('AUG_s42');d=r['datasets']['ETH']['AUG_s42']['decision']
c=next(x for x in v['conditional_selection'] if x['dataset']=='ETH' and x['model']=='AUG_s42')
lines=['# 减速诊断：结果解读','',
'实验已完成，8组保存输出全部纳入：ETH的B0、宽速度增强、T1，以及C0 dev的B0 seed42–46。正式计算耗时约2.5秒；实现检查、输入校验和独立逐窗复核另计。无新训练、无模型推理、无GPU使用。','',
'**主结论为B（选择问题），B1/B2暂不能确定。** 在ETH宽增强的89个方向有效减速窗口中，70窗存在纵向误差≤0.4m的候选，但67窗的Top1纵向过冲>0.4m。满足事先固定的B条件，不满足A。B的两种细分都不满足，按运行前补充规则保留B_UNRESOLVED，不能据此判为“都不明显”。','',
f"在确有纵向可达候选的{c['n_reachable']}窗中，{c['n_reachable_but_top1_overshoot']}窗仍选中过冲的Top1，占{c['top1_overshoot_given_reachable']:.1%}。这是同一窗口内的条件统计，比单独比较两个总体比例更直接。一维可达不保证该候选横向误差或完整轨迹也好。",'',
'| 指标：ETH减速89窗 | 原B0 | 宽速度增强 |','|---|---:|---:|',
f"| 纵向可达 | {b['t12']['longitudinal_reachable_rate']:.1%} | {a['t12']['longitudinal_reachable_rate']:.1%} |",
f"| Top1过冲 | {b['t12']['top1_overshoot_rate']:.1%} | {a['t12']['top1_overshoot_rate']:.1%} |",
f"| 全部候选过冲 | {b['t12']['all_overshoot_rate']:.1%} | {a['t12']['all_overshoot_rate']:.1%} |",
f"| 每窗减速候选平均数 | {b['mean_n_decel']:.2f} | {a['mean_n_decel']:.2f} |",
f"| 减速候选总概率P_dec | {b['mean_P_dec']:.3f} | {a['mean_P_dec']:.3f} |",
f"| FDE最优候选分数名次中位数 | {b['t12']['best_rank_median']:.0f} | {a['t12']['best_rank_median']:.0f} |",
f"| 最优候选纵向绝对误差均值（m） | {b['t12']['best_prog_error']['absolute_mean']:.4f} | {a['t12']['best_prog_error']['absolute_mean']:.4f} |",
f"| 最优候选横向绝对误差均值（m） | {b['t12']['best_lat_error']['absolute_mean']:.4f} | {a['t12']['best_lat_error']['absolute_mean']:.4f} |",'',
f"增强后减速候选数减少{1-a['mean_n_decel']/b['mean_n_decel']:.1%}，纵向可达率下降，最优候选的纵向和横向绝对误差都增大。说明生成质量也有退步；“未达到A阈值”不等于生成完全没有问题。候选数量和误差共同变化与这一退步相符，不能仅凭相关性确定唯一原因。",'',
'最优候选纵向带符号均值原B0约−0.041m、增强约+0.001m，虽然接近0，绝对误差并不小；此前仅看带符号均值确实掩盖了离散。这里89窗的minFDE是方向有效子集指标，不应与前次全部92窗的减速minFDE直接混用。','',
'历史中已经减速的15窗，增强Top1过冲率40.0%；历史平稳的62窗为91.9%。因此没有满足B1的过冲比例条件。对全部方向有效且历史减速的64窗，实际减速比例减去平均P_dec约−0.012，也不满足低估≥0.2的条件。前一个15窗是未来互斥减速组，后一个64窗是校准总体，不能混淆。','',
'85.1%的减速过冲案例来自历史平稳组，满足B2的集中条件；但ETH校准并未满足所有档偏差<0.1，所以也不能判为B2，更不能声称“减速本来就不可预测”。','',
'C0 dev五个seed都满足B主类，每个seed的有效减速组642窗：纵向可达97.7%–99.5%，Top1过冲52.0%–55.8%。seed42/43/44/46满足B2的操作性条件；seed45的高概率档偏差约0.181，保留B_UNRESOLVED。相同4753窗未重复合并为五倍样本。选择现象也存在于C0，ETH另有更明显的校准偏差和增强后候选退步，不能把它归为ETH独有，也不能作所有场景普遍成立的结论。','',
'下一步探索优先检查候选评分和概率校准，同时保留减速候选覆盖与横向误差作为约束；这比直接扩大速度增强或重跑五折更符合当前低成本探索目标。任何新打分规则应在独立开发数据上设计和选择，ETH继续只作已被反复查看的探索结果。本次没有启动下一项实验。','',
'文件：DECEL_DIAG.md为完整主表，DECEL_DIAG.json含全部机动×历史分层与t6/t12指标；populations/<数据>/<模型>/decel_window_metrics.npz为8份逐窗口派生结果，均不含原始坐标。INDEPENDENT_VERIFICATION.json记录对所有窗口、20候选、两个时点的原始预测独立公式复核，以及全部分层的样本数、校准、主要误差与比例和A/B判定检查。源文件和封存结果散列检查通过。','']
(ROOT/'RESULT_INTERPRETATION_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
print('INTERPRETATION_WRITTEN')
