# 评价、统计、停止标准与报告模板

状态：DESIGN_ONLY。以下新推进门属于本系列预先规定的研究标准；不替换历史major门，不把“可继续研究”改称“创新点已完成”。

## 1. 三种成功必须分开

1. **执行成功**：源、ID、实现、恢复和保存复算正确。
2. **可预测性信号**：显式历史H相对候选几何G确实提高当前目标预测，并能改善对应动作效用。
3. **部署推进证据**：固定K20获得稳定完整覆盖收益，且原成功与运动学全部过门。

不能因1成立推2，更不能因2成立推3。若1未通过，科学结果不可解释；若2未通过，不能声称历史无信息；若3未通过，不升级baseline。

## 2. 一级指标：是否有增量预测信息

主人口hold change，整体/稳定/过渡及inner为完整补充。所有seed分别报告，不把三个seed窗口拼成三倍样本。

每个分支预定两个主比较：

- `Delta_MSE = MSE_H - MSE_G`，在归一化安全教师参数上，先每窗平均20槽与2坐标。
- `Delta_regret = regret_H - regret_G`，使用guarded有限动作，先窗内平均20槽；J=同一候选max(ADE/.2,FDE/.4)。

可预测性信号门：两个差的3seed均值都<0、每seed均<0，且两项校正后的探索CI上界都<0。另要求H每seed的regret点估计低于identity，否则可能只是H比错误G少坏，并未利用可用机会。identity比较完整给描述CI，此附加点估计检查不冒称新的显著性检验，也不新增反复选主指标。

说明：参数MSE下降但效用regret不降，标 `PREDICTABLE_PARAMETER_WITHOUT_UTILITY_TRANSFER`；regret下降而MSE不降，可能多解标签不恰当，标 `UTILITY_SIGNAL_WITH_PARAMETER_AMBIGUITY`，不谎报两项都通过。这些分支可供下一轮重新定义问题，本系列不自动修改标签或多解阈值。

必须分层披露全部窗口/原成功/原失败、教师identity/非identity、精细模式诊断类别、终点可达/不可达、scene。后几项有GT选择效应，只是描述，不能替换完整人口主指标。不能用近并列参数命中率作为主要胜利。

## 3. 二级指标：是否改善完整候选集合

每端口/人口/group/seed保存：

- K20最小ADE、最小FDE及最小J；明确minADE/minFDE可能不来自同槽，coverage必须来自同槽good。
- Native→raw和Native→guarded的rescue/lost/retained、净增量和分母。
- GT-free全K20原SciPy complete-link：D=max(meanpath/.10,endpoint/.20)，cut1；effective-good、distinct及重复统计。有效good不是越多越好。
- 所有五pooled motion；逐候选和全20窗口通过率；junction mean/p95。
- 单J最优完整轨迹的早/中/晚误差，保留槽号，不拼接分段oracle。
- 执行非identity比例、raw→guard拒绝比例及各项原因；0改动明确记Native identity，不称安全改善。
- 有效参数量、训练/推理MAC口径、实际CPU/GPU计时、动作解码与安全检查耗时、峰值内存。Native缓存查表成本不能当完整F16运行成本。

P1/P2部署推进门（每分支独立）：

1. hold change的guarded对Native覆盖增益每seed>0，3seed平均≥2个百分点；校正探索CI下界>0。
2. hold与inner各group的原成功lost/native_good≤1%，每seed分别满足，零分母标NA并列实数，不用平滑分母。
3. hold与inner整体/稳定/过渡的minADE、minFDE相对Native均不恶化超过1%；change的ADE/FDE也不得恶化。
4. hold与inner每group每seed原五pooled motion均≤Native×1.01；新增junction mean/p95也≤Native×1.01。无floor/评价epsilon。
5. inner change每seed净覆盖不为负，避免只凭主hold通过忽略反例。

2个百分点是本系列的最低非零部署推进效果，不是重新定义历史major。正式宣称大突破仍需原change ADE与FDE各≥5%改善、保护1%、3seed一致、原joint CI/同路径覆盖/运动门及相关性充分性审核全部满足。边界病例与`.3/.5`敏感性完整另报，不替代`.2/.4`主标准。

## 4. 统计合同

- 重抽样单位为sequence|target整轨迹；保留该轨迹全部窗口。先在固定人口整体track框抽样，再应用group mask，明确frame track数与group support track数。
- 重复20000次，RNG seed=20261028；方法与Native/G/H共享同一track抽样索引。每次再配对重抽3个新模型seed；另给固定seed均值、只抽track的敏感性，避免3个seed抽样的离散性造成假稳定。
- 轨迹draw导致某group空时标invalid并报告，不补0；有效draw不足19000则该CI为未充分可估，不自动补采到好看。
- 家族固定为P1/P2各MSE、regret、coverage共6项；即使某分支未执行，也保留6项分母。双侧Bonferroni区间使用分位点0.05/(2×6)、1−0.05/(2×6)，同时给未校正95%描述CI。通过门使用校正区间；不从六项择最显著。
- P3为触发后的适应性组合验证：E+T相对E、相对T的覆盖差各列双比较校正区间，仍注明受前阶段选择影响，非独立确认性结论。若要认为组合更好，需两比较每seed正、各自均值≥1个百分点、校正下界>0，且所有保护/运动门继续满足。
- scene只有3个且分布不均，只给分母与描述反例，不造“跨场景总体独立显著性”。候选/20槽不是独立样本；Native多份seed缓存不是模型重复。
- P0是精确有限集合见证，置信区间只描述当前窗口人口，不称优化算法或预测器的泛化置信度。

## 5. 联合结果的解释

固定2×2端口记覆盖率C00、C10、C01、C11。报告交互项 `I=C11-C10-C01+C00`，并同时报告rescue集合重叠、lost、拒绝原因。

I为正不等于因果协同：训练是分开的，部署过滤非线性，研究数据反复使用。重点看C11是否确实优于两个单分支且全部安全保留门通过。不允许用“END单独救回+TIME单独救回”直接相加预测联合效果。

只有一个分支通过：交付该分支结果和另一分支停止理由；P3写NOT_TRIGGERED。两个都不通过：本系列结束，不能为了长时间自主运行再追加参数扫描。

## 6. 轻量化与总体研究含义

本系列模型每个4770参数，部署逐20槽共享推断，**不意味着整个系统就是4770参数**。F16主干、Native生成、后处理和安全检查都需要分别计入总系统成本。

可能支持的论文方向（必须有结果才能使用）：在固定候选预算下，以低维条件化运动过程修正补足Native精度，并显式审查成功保留及物理代价。不能仅凭设计将其写为已实现创新。

“历史+候选”H优于G只支持显式历史的增量可利用性；G本身的Native早已由历史模型生成。H不优于G不证明缺乏历史信息。全体拟合失败还可能来自小模型容量、参数多解、identity标签占优和回归均值化；这些替代解释必须写入结论，不能直接归罪于不可预测性。

目前不加邻居输入或scene特征，避免扩展已知有来源歧义的sidecar；也不能据本系列结论否认场景或交互信息的潜在价值。

## 7. 必须交付的文件

```text
README.md / protocol.json / source_manifest.json / split_manifest.json
P0/INPUT_AUDIT.json / ACTION_MANIFEST.json / GT_FREE_MASK_LOCK.json
P0/SAFE_OPPORTUNITY.csv / TEACHER_AMBIGUITY.csv / STAGE_DECISION.json
P1|P2/<arm>/<seed>/checkpoint.pt / run_manifest.json / history.json
P1|P2/<arm>/<seed>/predicted_parameters.npz / selected_actions.npz
metrics/{per_sample,groups,coverage,rescue_lost,motion,junction,cost}.csv
statistics/bootstrap_indices.npz / CI.json
completion_review/REVIEW.json / SOURCE_PRESERVATION.json
RESULT_ANALYSIS.md / RESULT_ANALYSIS.json
```

每个checkpoint绑定source/split/statistics/动作合同/init/optimizer/RNG及8epoch history。保存足够参数与Native引用SHA，以独立重建raw与guarded路径；若只存指标无法重建，不算审核完整。压缩/去重不得删除旧唯一证据，存储不足先说明具体缺口。

CPU独立审核全部保存动作重构、同槽ADE/FDE/J、coverage、full complete-link、motion/junction、CI与拒绝原因；模型恢复检查与算术审核是两件事。不将同助手第二份实现称外部独立评审，不冒称再次回放全部模型梯度。

## 8. 最终报告必须回答

1. END49/TIME441在安全与junction过滤后还剩多少机会？与旧GT oracle为何不同？
2. tiny拟合、全train、val、hold、inner分别如何？没有通过哪一道门？
3. H相对G是否有参数和动作效用增量？是否优于identity？
4. 是否只是更多输出identity而显得安全？是否只是修好了已经good的样例？
5. 哪些旧成功被破坏？raw和guarded是否掩盖了模型原始运动代价？
6. 是否得到固定K20完整覆盖的稳定收益？新增方向与时间模式是否真正帮助同一完整轨迹？
7. 若联合，是否真的超过两个单分支，而非GT拼接或重复计算rescue？
8. 哪些项目STOP/NOT_TRIGGERED？为什么？不能把未执行写为阴性实验。

## 9. 预先反方审查（同助手，非外审）

- **监督多解**：最优动作可以不唯一，单参数回归可能学均值。用tie/近等效集合直径、参数误差与效用regret双证据限制结论；本轮不按结果换loss。
- **过强保守过滤**：逐候选motion+junction可能删掉大量真实可行的整体改进。P0同时显示每道过滤损失，停止只针对本合同，不证明不存在安全运动生成。
- **全槽同GT收缩**：每个候选都朝同一GT优化可能减少互补性。保持K20完整聚簇/有效good与原成功损失统计；oracle标签正确不等集合训练正确。
- **研究数据复用**：hold已有多轮结果暴露。此系列是机制筛选，不能作为最终论文独立验证；后续确认性设计另行确定，不在此擅用新fold/test。
- **算术与物理不等价**：数值容差仅作复算比较，运动门literal不变；未来内部差分通过不代替历史衔接，逐候选p95不代替pooled p95。

这些风险已经纳入停止规则，不靠增加任务时长解决。本次仅设计交付，无已完成预检、性能提升或新启动声明。
