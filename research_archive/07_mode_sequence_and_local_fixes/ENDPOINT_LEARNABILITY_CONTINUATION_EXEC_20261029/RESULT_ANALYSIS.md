# END安全端点修正续执行：tiny V2已完成，未过固定拟合门

## 结论

已按人类“按照文档执行实验”落实V2抽样、来源复用、当前小批次桥接及唯一H seed42 tiny拟合。固定256次更新后，非identity候选MSE下降 **80.3173%**，未达到原90%门，登记 **STOP_TINY_FIT_V2**。

独立CPU保存审核通过，执行完整性与科学门状态分开：保存计算正确，不代表拟合门通过。按文档到此停止，**六个正式G/H模型未启动，hold/inner GT本轮未开放，H是否优于G尚未检验**。TIME停止、P3不触发；不追加网络、epoch、LR、loss或重新抽样。

## 1. 实际实施与来源

- 新目录ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029，5140来源／依赖条目，继承原4451来源及learning_source_extension和684个旧不可变输出的原digest；全18核心源码SHA与AST完全一致。旧P0、GPU8与真实513/12两epoch全RNG恢复标INHERITED_EXACT_DEPENDENCY_MATCH，没有冒称本轮全部重跑，也没有重复P0枚举。
- 当前新跑真实8窗G/H桥接：旧／新小网络输出、loss、全部参数梯度bitwise，优化器更新0。其作用是版本依赖桥接，不是新增正式模型。
- V2原hash前64活跃窗保留，后64从剩余train同hash取，128个不同train窗、97条track；当前所有窗活跃，不称正负平衡。2057/2560候选活跃，503个identity标签仍保留。SELECTION在任何tiny更新前封存，源SHA和初末检查完整。
- 首次调用仅因旧审核helper的p0_driver兄弟模块导入路径缺失而在创建模型前失败，0参数更新。原失败、锁源与代码保留；新namespace-only适配器追加原ROOT，调用同一冻结run_tiny，无数学／采样／超参变化。本轮只有一个真正训练的H42 tiny，而非挑过门的重试。

## 2. 固定tiny：参数拟合与动作效用

| 量 | 初始0步 | 最终256步 |
|---|---:|---:|
| 非identity候选MSE，2057候选 | 0.377369956 | 0.074276728 |
| 全体候选MSE，2560候选 | 0.303222656 | 0.072455944 |
| identity候选MSE，503候选 | 0 | 0.065009912 |
| guarded精确教师动作命中率 | 19.6484% | 30.2734% |
| guarded平均J-regret | 0.565533311 | 0.194834317 |
| raw平均J-regret | 0.565533311 | 0.157932124 |

identity初始MSE为0，只给绝对变化，不计算无定义的相对下降。主90%门严格使用原float32训练target→float64误差口径；报告标签float64误差的微小舍入差仅为算术复核，不进入good／motion科学门。

固定budget=128窗、batch32、64轮、256updates，4770参数、H、seed42、原AdamW .001/wd.0001/clip5、末层零初始化。训练更新部分实测4.7193s；这不是含source SHA、IO、保存审核的总耗时或正式六模型ETA。

## 3. tiny训练子集的完整K20质量

| 最终端口／组 | N | Native好窗 | 好窗 | 救回 | 丢失 |
|---|---:|---:|---:|---:|---:|
| raw/overall | 128 | 69 | 67 | 8 | 10 |
| raw/stable | 79 | 47 | 48 | 6 | 5 |
| raw/transition | 26 | 13 | 10 | 1 | 4 |
| raw/change | 23 | 9 | 9 | 1 | 1 |
| guarded/overall | 128 | 69 | 70 | 8 | 7 |
| guarded/stable | 79 | 47 | 50 | 6 | 3 |
| guarded/transition | 26 | 13 | 11 | 1 | 3 |
| guarded/change | 23 | 9 | 9 | 1 | 1 |

guarded整体69→70，仅净增1/128；救回8但丢7，原成功丢失7/69=10.1449%。这只是本tiny训练子集的描述，不冒充hold保护门统计或泛化证据。change仅23窗，9→9、救回1/丢1，不用这个小分母作主要结论。

guard拒绝398/2560候选（15.5469%），speed p95失败153、junction失败270，其余五个失败位0；可同时失败，不能相加当互斥原因。执行非identity约78.36%。raw整体67窗（救回8/丢10），guarded恢复其中3个原成功，但并未保证GT成功保留。

最终guarded四组的原五pooled motion及junction mean/p95均过1.01门，**物理门通过仍不等于准确性保护通过**。整体minADE从0.172393743变0.181194657、minFDE从0.304169528变0.318398568，连续误差仍有退化。单槽平均参数误差／regret下降与集合minJ、完整coverage不是同一目标，完整原数组和所有分组已保存。

## 4. 一次有限原因核对（不是唯一根因判断）

1. 拟合：active MSE已下降80.32%，因此不是完全没有学习；但当前固定映射／预算尚未达到预定90%tiny可拟合门。不能从这一个H训练子集判断输入信息不存在。
2. 参数与决策：精确动作命中仅约30.27%，回归平均误差下降不自动跨越离散动作边界，也不一定命中同样有效的其他动作。不能以命中低直接判全部动作无效。
3. guard：拒绝约15.55%，使guarded regret高于raw，符合安全过滤会放弃部分误差收益；guard同时减少了3个lost，不能只把它称阻碍。
4. 保留：503个identity候选从0误差变为正误差，存在对原不应改候选的偏离；但候选标签identity与原成功窗口不是同一人口。不能将这一个现象直接定为唯一损失根因或自动增加新protection loss。
5. 集合效用：guarded regret从.5655降至.1948，但完整好窗仅净增1且丢失7。说明“平均校正变准”不足以证明完整K20预测已稳健改善；并不否定所有端点校正方案。

本轮没有继续拟合替代模型、没有换128窗、没有把H−G主假设换成tiny−Native。下一轮方向需要基于这些保存证据另立合同，本轮不自动追加。

## 5. 审核与未执行项目

- completion_review/TINY_REVIEW.json：初始化/最终、raw/guarded共512个window-port、10240个candidate-port，连续参数→最近动作、raw/guarded几何／literal失败位、同槽ADE/FDE/J、原SciPy完整K20 complete-link独立condensed-pair重算、effective-good/distinct、全组pooled五motion、source初末SHA。
- checkpoint独立CPU读取：64history/256updates、全4770参数finite、fresh初值、输入／target digest、source与selection绑定、AdamW state步数256和moment finite、全部RNG payload存在。没有第二次forward/backward或optimizer replay，不是外部评审。
- PREFLIGHT.json逐项区分旧证据继承、本轮新跑和tiny停止后未执行；不是生产总PASS。正式六模型、24模型人口、hold/inner新teacher、主family6/20kCI、全模块延迟成本均NOT_RUN，不能伪造性能／显式历史增量／部署结论。
- 资源沿用v3预警优先；实测峰和最大载荷列在RESOURCE_INVENTORY/RESOURCE，未恢复机械6GiB、未OS/pagefile更改或kill用户程序；未删除文件腾空间。旧automation-4与全部旧PAUSED未修改；本轮无新OS任务或自动记录需要删除。

## 6. 同助手反方DA1／DA2／DA3（非外部评审）

- DA1：V2在已见分层不足后修订，明确不是原双分层的无偏复现；128IDs在更新前固定且不以拟合效果挑样，得到人类新文档执行授权。G仍含已冻结历史信息，H/G的命名不能改写为有／无全部历史。
- DA2：最强反证是参数MSE和regret大幅下降仍伴随lost与连续误差退化。不能只展示80.32%下降和净增1而隐藏丢7，也不能把tiny物理门PASS称部署安全成功。
- DA3：tiny未过不证明历史不可观测；tiny无G对照、只有一个训练seed、没有held结果。正确结论是当前固定设置未满足进入正式实验的门槛。END是有限轻量位移／平均速度偏差校正，不是新停走／意图生成器。TIME/P3停止及原major／保护门均不放宽，baseline/historycycle11/长期目标不假完成。

报告由AI助手辅助实施和同助手审核。原P0及所有来源、失败、模型、输出和日志保留；未安装依赖、未改旧环境、未查询旧completed PID/task/log。
