# W0 结果与反方审核

2026-10-18。状态：ANALYZED；NONFORMAL oracle 诊断，不增加正式训练计数。原设计七份文档和旧F0结果不改。

## 结论

**WINDOW_WITNESS_PRESENT=true；PORTFOLIO_GLOBAL_SAFE=true。** 原固定32个native失败窗口中，13个找到真实生产精度下合格安全修正。17个目标slot方案通过全部严格门，全部达到预注册robust_margin≥1e-4。由此支持“这批已知GT样本在当前U0域内存在安全改善机会”，不支持“已经学会从past修正或迁移”。

这与旧F0 accepted0不矛盾：W0联合改变求解目标为max gamma及内部数值余量/取消事后shrink，不是重写F0或只改评价epsilon，也不是纯单因素目标因果比较。

## 执行与审核事实

- 当前worker32008/service2804：00:15:31.359485→00:17:38.194564 HK，双SUCCESS，约126.835秒。原Windows瞬时写入失败版本23860/所有代码日志和46题证据保留；46题通过SHA/原科学代码兼容独立核验复用，剩594题新求解；总640，不是重新求解全部。
- 本次人类10:10:05.4868695的唯一迟到完成核验见Ready/processnone/XML0/stderr0/640。此前未完成恢复版startup/半程记录，**不伪称按时半程或startup已执行**。独立OS工作已在对话空闲期间完成。
- 源7580、digest `a53ecd673f65791e2a32a6e9d291d4c3b0fcece5ddfe17c643532c998cc6ae9e`；1287输出SHA、ARITHMETIC_AUDIT、PRESERVATION、RESOURCE全部PASS。原F16/decoder/scorer未加载/训练，旧source/model/result未改。
- [completion_review/REVIEW.json](completion_review/REVIEW.json)：额外CPU复核226.314秒，640题raw=atanh(primal)转float32逐项EXACT（无post-shrink），独立controls递推/余量/标签/最大margin后最小物理slot选择、全FIT原SciPy完整K20聚簇和四组five-motion，最大算术差2.220446049250313e-16；全部source/output前后SHA不变。无Torch/模型/梯度/optimizer/solver重放；同助手第二保存算术实现，非外审。

## 求解状态与严格分类必须分开

| 类别 | 数量 |
|---|---:|
| 全部attempt | 640 |
| infeasible | 622 |
| optimal / 可有限atanh重构 | 18 / 18 |
| 原始primal所有内部约束literal通过 | 0 |
| 实际完整候选质量+位置+窗口全部五motion严格通过 | 17 |
| 其中robust_margin≥1e-4 | 17 |
| 严格通过覆盖的独立注册窗口 | 13 / 32 |
| 有optimal但实际五motion拒绝 | 1（sample22057/slot7，accel_p95） |

原始primal会在求解器容差级别违反一项或多项内部约束，**不能称精确凸可行解或数学安全证明**。按预注册规则，只要各分量abs<1能有限atanh，实际U0映射仍可作为提案；最终科学接受只看保存真实路径的原literal门，未加epsilon。18题均可重构；17题实际通过，1题虽质量margin约2.6996%仍因accel_p95失败而拒绝。未对p95做假凸约束。17题actual normalized margin约2.517974%–49.769764%，明显大于1e-4标签门，不是将1e-8误差四舍五入为成功。

640题是32个条件失败窗口×20slot，不是640独立样本；无训练seed，也不补置信区间制造总体代表性。622个solver infeasible不排空原pooled安全域。

## 完整FIT oracle portfolio

每失败窗口从严格通过方案中按最大actual margin、原slot最小破并列；所有其他窗口native identity，16个成功对照亦identity。**这个使用GT的选择不能部署或用作P1 teacher/owner。**

| group | N | native good | portfolio good | rescue | lost | 净覆盖增量pp | 五pooled motion |
|---|---:|---:|---:|---:|---:|---:|---|
| overall | 18909 | 10858 | 10871 | 13 | 0 | 0.068750 | PASS |
| stable | 10767 | 7414 | 7414 | 0 | 0 | 0.000000 | PASS |
| transition | 3417 | 1507 | 1507 | 0 | 0 | 0.000000 | PASS |
| change | 4725 | 1937 | 1950 | 13 | 0 | 0.275132 | PASS |

完整FIT change提升仅13/4725=0.275132pp，overall约0.06875pp，不是重大门5pp。稳定/过渡保持、原好窗口lost0在此部分来自预先identity，不能称新head学会保留。可同时修改20条并分配运动学预算，不应解释为只改目标slot且19条不动。

完整每组native与portfolio minADE/FDE/J、effective-good/distinct、同Jslot早中晚、五项绝对motion和rescue/lost IDs在RESULT.json及本报告JSON完整保留；五项包含speedp95、accelmean/p95、jerkmean/p95，dt=.4。窗口通过与全FITpooled组合通过分别验证，后者不是由前者自动推出。

### 13个被选方案（slot为0起始物理序号）

| sample ID | slot | actual margin |
|---|---:|---:|
| 22199 | 3 | 40.552559% |
| 26182 | 6 | 21.031947% |
| 11514 | 4 | 30.731997% |
| 20240 | 10 | 13.502713% |
| 23344 | 0 | 14.252901% |
| 14247 | 15 | 26.352653% |
| 11234 | 19 | 6.894660% |
| 20901 | 14 | 46.431337% |
| 22731 | 15 | 4.436462% |
| 2415 | 7 | 2.517974% |
| 2998 | 16 | 12.798089% |
| 1147 | 14 | 24.780433% |
| 10242 | 2 | 49.769764% |

## ARS inline DA2：证据综合反方（同助手，非外审）

最强问题：GT辅助求解、按GT margin选择，再用GT评价，当然比学习型部署容易。回应：结论限于存在性；不提供P1训练标签/超参，不将13/32条件比例推广到全部失败窗口。旧F0 accepted0不改，I1均值回归塌缩、T1保留—救回折中、T2迁移和motion失败继续构成反证。**判定：解释PASS_WITH_LIMITATIONS，不是学习机制PASS。**

## ARS inline DA3：最终压力检查（同助手，非外审）

最强反驳：所有18原始primal内部literal检查均未通过，如何谈安全？区分：不声称这些primal的精确凸可行性；真正被计数的是有限raw经原float32 tanh/单位盘和double积分产生的实际路径，经原严格质量、位置与五运动门独立算术再次核验。该实际见证存在性仍成立，但数值结果不是形式化证明。13个选择依赖GT、已见FIT、窗口/场景相关，不等独立泛化/心理意图/唯一decoder或sharedhead根因。**判定：有限实证见证PASS；learnability/generalization/major均未证明。**

## 下一步与停止

W0一次640题结束，不再W1/W2/solver/余量扫参，旧G1保持NOT_TRIGGERED。本独立P1已于10:14:58.651415开始四臂真实学习；W0阳性不改变它的owner、loss、数据或门。P1四臂全部保存审核/预注册family24及native3/9后，仅D满足全部机制+保护+motion门才P2；否则结束有限系列。不提高baseline，不重置historycycle11/总goal。

