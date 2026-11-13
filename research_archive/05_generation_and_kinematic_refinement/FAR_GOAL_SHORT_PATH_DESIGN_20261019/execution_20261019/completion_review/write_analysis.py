"""Assemble evidence-grounded report from the completed CPU review; no training."""
import sys,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(H));import common as c
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    review=read(H/'completion_review/REVIEW.json');assert review['status']=='PASS_COMPLETION_SAVED_SUMMARY_REVIEW'
    r=read(H/'RESULT.json');records=r['records'];rows=review['normal_all96_group_rows'];h2=review['H2_all216_group_rows'];boot=review['primary_full16'];H0=read(H/'LONG_LABEL_AUDIT.json')
    def rec(a,s,p,g):return records[f'{a}_seed{s}/normal/{p}']['metrics'][g]
    def base(p,g):return records['native/normal/'+p]['metrics'][g]
    def counts(a,p,g):return [rec(a,s,p,g)['good'] for s in (42,43,44)]
    means={}
    for p in ('FIT','outer','inner','original_CAL'):
        for a in ('A','B'):
            means[a+'/'+p]={}
            for g in ('overall','stable','transition','change'):
                vals=[rec(a,s,p,g) for s in (42,43,44)];native=base(p,g)
                means[a+'/'+p][g]=dict(n=native['n'],native_good=native['good'],good=counts(a,p,g),mean_good=float(np.mean(counts(a,p,g))),delta_coverage_pp=100*(np.mean(counts(a,p,g))-native['good'])/native['n'],minADE=float(np.mean([v['minADE'] for v in vals])),minFDE=float(np.mean([v['minFDE'] for v in vals])),J=float(np.mean([v['J'] for v in vals])),ADE_harm_pct=100*(np.mean([v['minADE'] for v in vals])/native['minADE']-1),FDE_harm_pct=100*(np.mean([v['minFDE'] for v in vals])/native['minFDE']-1))
    wall=(__import__('datetime').datetime.fromisoformat(read(H/'exit_status.json')['finished_at'])-__import__('datetime').datetime.fromisoformat(read(H/'worker_state.json')['started_at'])).total_seconds()
    text=['# 远期目标条件—原12步候选：完整结果分析','',
          '状态：COMPLETED_WITH_RESULTS；H0/H1/H2及保存审核均已完成。科学结论：远期目标优势未建立，MAJOR_FAIL，不升级baseline。','',
          '## 结论先行','',
          'goal20相对goal12在outer变化轨迹上三个seed均有正向点估计，但主校正区间跨0；两者仍不如native。新生成器明显使用目标，并非把目标完全忽略。然而更远目标尚未转化为可靠的原12步完整候选改善，且存在严重成功丢失和平均加速度代价。',
          '',
          '**当前应走的分支：目标使用已经证实有明显响应，但远目标→短期路径的有效映射和迁移尚未解决。** 不把结果归结为“只需预测更远”“process完全忽略目标”或“只需scorer”。目标预测和过程表达仍是耦合的候选瓶颈，当前证据不唯一识别根因。','',
          '## 1. 执行与证据边界','',
          f'- worker/exit双SUCCESS：2026-10-19 21:48:27.377703至22:30:40.241109HK，实际{wall/60:.4f}min。6fresh模型、150epoch、11100optimizer steps；82保存任务=24正常+54冻结干预+4native参考。',
          '- 7505源初末SHA、270输出文件（含6run模型/合同/历史）、全部82保存质量/原GT-free全K20聚簇/五pooled运动学独立审核和20k bootstrap全draw独立复算PASS。执行PASS不是科学效果PASS。',
          f"- 本次补充CPU保存审核{review['seconds']:.3f}s，全部82质量/计数/zabc路径重构、H2override/donor及同ID人口、16主点估计/全部6history复核PASS，最大保存算术误差{review['max_independent_geometry_quality_error']:.6g}；7505source/270output SHA前后不变。补充没有再次完整重跑聚簇/bootstrap，更没有模型前向/反向/训练。",
          '- 原worker审计为完整保存算术复算；本报告分析和ARS反方为同一助手完成，非独立外部评审。所有原RESULT、源、模型、失败版本不改。',
          '- 未发现保存的halfway_check，不宣称按时完成22:07半程检查；本轮由人类“分析结果”直接触发已完成产物分析，没有查询任何已结束PID或Windows任务。','',
          '## 2. H0标签合同与选择效应','',
          '- FIT18909中m20有效13856/523tracks，change4725中有效3898/269tracks；outer1035有效764，outer change253有效208/60tracks；inner1856有效1362；CAL2879有效2177。',
          '- FIT432个新增未来13..20与保留目标区间相交，仅关闭aux。101个FIT时间歧义、4520个FIT缺长标注同样保留短期训练及ALL评价。原28791IDs全登记，科学四人口24679，不删除短标签难例。',
          '- 有效FIT速度均值.4787m/s、native覆盖60.65%；缺标签/被mask群.8471m/s、覆盖48.58%。长标签可用性确有选择效应；因此ALL而非m20子集是主分母。A和B使用相同m20、同批次分母。',
          '- 两臂固定原8步/P2，未因20步标签重选邻居；旧12步未来参与邻居选择的legacy偏差并未消除。target-disjoint不等于context-disjoint，原CAL已反复使用。','',
          '## 3. 原12步完整覆盖','',
          '同一条candidate同时ADE≤.2且FDE≤.4；下面是成功窗口数，非minADE/minFDE跨slot拼接。seed顺序42/43/44。','',
          '| 人口 | N | native | goal12 A | goal20 B | B−native平均覆盖(pp) |',
          '|---|---:|---:|---|---|---:|']
    for p in ('FIT','outer','inner','original_CAL'):
        for g in ('overall','change'):
            x=means['B/'+p][g];text.append(f"| {p}/{g} | {x['n']} | {x['native_good']} | {'/'.join(map(str,counts('A',p,g)))} | {'/'.join(map(str,counts('B',p,g)))} | {x['delta_coverage_pp']:+.3f} |")
    b=means['B/outer']['change'];a=means['A/outer']['change']
    text+=['',f"- outer change：A平均{a['mean_good']:.3f}/253，B平均{b['mean_good']:.3f}/253，native110/253。B−A平均+8.667窗、+3.426pp；B−native平均−6窗、−2.372pp。每seed相对native均下降，而不是达到本系列每seed至少+2pp（净+6窗）的门。",
           f"- B的change ADE/FDE相对native平均恶化{b['ADE_harm_pct']:.3f}%/{b['FDE_harm_pct']:.3f}%，并非达到各5%改善。FIT覆盖改善而outer/CAL的change仍下降，是迁移不足的描述性模式；这里不把一组显著另一组不显著当成组间差异检验。",
           '- 不隐去反例：outer overall的B为635/624/630，相对native625有正有负，平均仅+.451pp；inner A的change172/172/166是正例。远目标B并未在所有人口优于A。','',
           '## 4. 原成功保留：救回不等于净收益','',
           '| outer change/seed | A救回/丢失 | B救回/丢失 | B丢失/native110 |',
           '|---|---|---|---:|']
    for seed in (42,43,44):
        aa=rec('A',seed,'outer','change');bb=rec('B',seed,'outer','change');text.append(f"| {seed} | {aa['rescued']}/{aa['lost']} | {bb['rescued']}/{bb['lost']} | {100*bb['lost_over_native_good']:.3f}% |")
    text+=['','B三个seed均丢失29/110个原成功窗口，即26.36%，远超1%保护门。B确实比A少丢一些，也多救回一些，但仍是救回22–24、丢失29，不能当成功生成。详细rescue/lost ID原样保存在各job metrics和原NPZ；A→B的churn另存于结构化报告，不混用两个不同基线。',
           '', '## 5. 主family16统计','',
           '20000 paired target-track×seed bootstrap，overall191tracks/change68tracks；全部20000有效。校正分位[.0015625,.9984375]；coverage为百分点，其余原单位。下表包含全部16项，不缩家族。','',
           '| 比较/人口/指标 | 点估计 | family16校正CI |','|---|---:|---|']
    for k,z in boot['metrics'].items():
        scale=100 if k.endswith('/coverage') else 1;lo,hi=z['CI_family16'];text.append(f"| {k} | {scale*z['point']:+.6f} | [{scale*lo:+.6f}, {scale*hi:+.6f}] |")
    text+=['','- B−A change coverage +3.426pp，校正CI[−3.316,+9.899]pp；即使普通95%CI[−.718,+7.535]也跨0。ADE/FDE/J的B−A校正区间同样跨0。**三seed同向是点估计证据，不等于不确定性已排除。**',
           '- B−native change ADE/FDE/J的校正CI全部在0以上，支持当前测试人口中误差恶化；coverage校正CI[−10.143,+4.490]pp仍宽，不能声称覆盖下降已独立显著。',
           '- B−native overall FDE和J校正区间也在0以上；overall覆盖小幅净增不能抵消这些反证。',
           '- 等track权重敏感性：change的B−A仍为+3.226pp，但overall B−A从窗口加权+.129pp变为−.544pp，提示overall小增益依赖权重。敏感性不替换预注册主统计。少scene/3seed/重复outer的探索性限制不变。','',
           '## 6. 运动学：不能把低jerk称为安全','',
           '原口径prepend最后观测原点0，dt.4，12v/11a/10j；全部window×K×time直接pool。literal≤1.01native，不加epsilon。表为outer change各seed相对native变化范围：','',
           '| 方法 | speed p95 | accel mean | accel p95 | jerk mean | jerk p95 |','|---|---:|---:|---:|---:|---:|']
    for arm in ('A','B'):
        rr=[z for z in rows if z['arm']==arm and z['split']=='outer' and z['group']=='change'];values=[]
        for k in ('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95'):
            v=[z['motion_pct'][k] for z in rr];values.append(f'{min(v):+.2f}%…{max(v):+.2f}%')
        text.append('| '+arm+' | '+' | '.join(values)+' |')
    text+=['','A/B各48个split×seed×group的五项AND门全部失败，不能误写成每项运动指标都恶化。这里p95和jerk普遍显著下降，但平均加速度持续过门；低阶平滑路径并不保证平均加速度不高。B outer change accelmean+29.74%至+34.53%，不是1%内安全成功。',
           '- 保留门A/B各48/48失败；protected overall/stable/transition误差门A17/36、B22/36失败。B的protected coverage门也有1项失败（outer overall seed43）。不得只报加速度或只报覆盖净值。','',
           '## 7. H2：模型有使用目标，但更远的真目标也不能补全短程形状','',
           '以下GT目标诊断严格在同m20=true change cohort比较，不与ALL分母混比：','',
           '| 人口 | m20 change N | A正常→GT12 | B正常→GT20 |','|---|---:|---|---|']
    for p in ('outer','inner','original_CAL'):
        cells=[];n=0
        for arm in ('A','B'):
            rr=[z for z in h2 if z['arm']==arm and z['split']==p and z['condition']=='GT_goal_local_diagnostic' and z['group']=='change'];rr.sort(key=lambda x:x['seed']);n=rr[0]['n'];cells.append('/'.join(str(x['normal_same_cohort_good']) for x in rr)+' → '+ '/'.join(str(x['intervention_good']) for x in rr))
        text.append(f'| {p} | {n} | '+ ' | '.join(cells)+' |')
    text+=['','- outer ALL change正常A=96/96/94，替换CV目标降到11/14/14，错配不同track目标降到43/44/47；B正常105/104/103，CV降到15/15/18，错配降到65/62/55。错配不使用GT选donor。',
           '- B outer换目标平均路径移动：CV .581–.599m；跨track .833–.864m；终点分别1.459–1.500m和2.138–2.219m。响应不是“几乎不变”。',
           '- B正常a范数约.231–.253m，A约.064–.070m。实际目标干预的a变化沿目标变化方向的抵消投影B约5.3–8.7%，A约1.4–2.0%；不支持“a把z完全抵消、模型忽略目标”。这只是固定模型局部响应，不是独立因果归因。',
           '- 同208个outer change长标注窗，A真实12步目标由85/84/81提升至172/172/171；B真实20步目标只由91/93/92到108/102/102。B的理想远目标仍有局部收益，但远小于“保证短期正确”：远期位移不能唯一决定原12步的停止、速度和转向；当前process/basis适配亦可能限制收益。',
           '- 不将A的GT12收益直接当“预测器可学上限”，不把GT20负例证明所有远目标无用。GT override会改变输入分布和多样性，且不是部署/严格上界。A/B物理目标时间不同，不能直接拿米误差大小宣称谁更准确。',
           '', '## 8. 训练内变化与有效候选','',
           '- 两臂所有seed训练损失均下降，末epoch没有完全死的slot，winner熵约2.73–2.75。不能据此宣布全部slots学习到可迁移运动规律；history是更新前batch加权统计，不是final population loss。',
           '- outer change A distinct约19.988–20，B约19.976–20，未发生20条完全复制成一簇；但更多distinct不等于更多有效good。B有效good簇均值约.502–.522仍不能补救覆盖/保留/运动门，不能靠复制或扩大K凑收益。',
           '- 全部seed/split/四group、m20与scene分层见原RESULT.records；结构化RESULT_ANALYSIS.json包含全部96正常组、216H2组和A→B churn，不只保留有利例。','',
           '## 9. 反方审核与最终分支','',
           '### DA2：结果综合反证（同助手，非外审）','',
           '最强支持：B相对A三个seed change覆盖都增加，FIT也提升。最强反证：CI跨0、B仍低于native，change ADE/FDE/J校正区间支持恶化，native-good丢失26.36%，motion全部组AND失败。不能以相对胜出升级baseline。缺长标签选择偏差由共同mask和ALL控制，但旧F16已见、context相关、远目标难度不同等替代解释仍在。结论保留“弱相对趋势”，不说无效等价。','',
           '### DA3：解释与行动边界（同助手，非外审）','',
           '最强反驳是“目标预测差而不是远目标本身无价值”；H2确实显示理想远目标能改善部分窗口，所以不能排除该解释。另一方面真远目标仍留下大量短期失败，而目标替换产生大响应且抵消不强，因此不应把一切归为“完全不使用目标”或只修goal predictor。当前只支持目标—短程过程映射/迁移仍未解决，不唯一认定表示、basis、优化或信息不足。','',
           '**最终分支：目标使用明显，但原12步完整候选改善与迁移未建立。** 下一大方向建议优先研究能约束短期运动过程的状态/时间结构条件，并明确与“更远终点”信息区分；这需要单独精确设计。本有限系列结束，不自动延长H、扫aux权重/basis/epoch，不启动scorer或旧模块解冻。baseline/historycycle11/unfinishedgoal不变。','',
           '## 10. 审核来源','',
           '- [原RESULT](RESULT.json)、[16项主比较](PRIMARY_COMPARISONS.json)、[完整保存算术](ARITHMETIC_AUDIT.json)、[bootstrap审核](BOOTSTRAP_AUDIT.json)。',
           '- [本次CPU复核](completion_review/REVIEW.json)、[复核源码](completion_review/review_results.py)、[H0标签审计](LONG_LABEL_AUDIT.json)、[源保护](PRESERVATION.json)、[实际资源](RESOURCE.json)。',
           '- 全部来源为本地保存证据；未使用外部文献制造支持，未重新训练，未进行独立外部审稿。',
           '']
    body='\n'.join(text);(H/'RESULT_ANALYSIS.md').write_text(body,encoding='utf-8')
    analysis=dict(status='ANALYZED',execution_status='COMPLETED_WITH_RESULTS',scientific_verdict='FAR_GOAL_SIGNAL_NOT_ESTABLISHED_MAJOR_FAIL',branch='TARGET_USED_SHORT_PATH_MAPPING_AND_TRANSFER_UNRESOLVED',at=c.now(),actual_worker_wall_seconds=wall,mean_normal_metrics=means,major_success=False,far_goal_signal=False,review=review,original_result_SHA=c.sha(H/'RESULT.json'),report_SHA=c.sha(H/'RESULT_ANALYSIS.md'),next_automatic_training=False,baseline_history_goal_unchanged=True,inline_DA2_DA3_not_external=True)
    c.write(H/'RESULT_ANALYSIS.json',analysis)
    print(json.dumps(dict(status='ANALYZED',wall_seconds=wall,report=str(H/'RESULT_ANALYSIS.md'),B_outer_change=b),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
