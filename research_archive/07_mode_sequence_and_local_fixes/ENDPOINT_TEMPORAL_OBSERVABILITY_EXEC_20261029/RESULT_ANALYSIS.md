# 端点／时间可观测性：已完成P0与端点实际GPU预检，等待tiny抽样合同澄清

## 当前结论

P0全部保存结果及独立复算通过。端点动作类有足够安全oracle机会；时间动作类未过原定研究价值门，停止该分支。
端点的GPU梯度和真实513/12恢复核验均通过，但tiny原实现要求的非活跃窗口层为空，故未执行tiny更新，也未放行正式训练。
这不是“模型拟合失败”，更不是“历史没有信息”。本系列尚未获得正式G/H对比或held覆盖改善证据。

| 分支／人口 | 原好窗 | 几何oracle好窗 | 加内部motion过滤 | 再加junction过滤 | 安全救回／轨迹 |
|---|---:|---:|---:|---:|---:|
| END / train (721 change窗) | 283 | 550 | 510 | 417 | 134 / 78 |
| END / val (256 change窗) | 113 | 195 | 183 | 157 | 44 / 23 |
| TIME / train (721 change窗) | 283 | 333 | 307 | 288 | 5 / 5 |
| TIME / val (256 change窗) | 113 | 130 | 120 | 113 | 0 / 0 |

训练change分母721，准入要求至少15救回且至少5条轨迹。END为134/721（+18.5853个百分点），TIME仅5/721（+0.6935个百分点）。验证集不用于调门，END验证44/256（+17.1875个百分点），TIME为0/256。所有oracle零lost由identity合法、同槽J最小化构造保证，不能当作学会保留原成功。

## 真实执行与审核

- 新执行锁4451来源，完整继承原4411旧digest；原18源码保留，新版本只批量化动作插值。实际hash128：END125440/TIME1128960动作与原实现路径逐位一致；独立几何最大误差0/1.776e-15，literal七失败位及教师选择EXACT。
- 全量独立复核40140800动作、8192个branch-population重复窗（4096唯一opt窗）、12个oracle人口／48组：同候选ADE/FDE/J、全部mask／tie／直径、K20 complete-link聚簇及有效good、五pooled motion和junction。安全oracle16组全部原五motion和新增junction人口门通过；不把逐候选通过逻辑上等同pooled门。
- END各候选可用动作平均train12.7378/val13.2023；train有50087/62060候选非identity，但3103/3103窗口至少一候选非identity。TIME的多解/tie和退化需完整保留，详见P0_AMBIGUITY.csv。
- 前台通道中断留下5个完整阶段和29个partial标签分片。原文件保留，在新命名空间逐字节复用，原冻结SHA、copy收据及全数学复算均通过。不是重做已完模型或删除失败。
- 独立CPU续算+审核worker 22:08:45.594371至22:19:10.310898双SUCCESS，约624.423秒；这不含此前前台与128预检，不冒称P0总耗时。唯一半程核验22:20:04.5588881发现已完整结束，不再查旧PID／task／log。
- 实际CUDA G/H各8窗分阶段梯度、零输出identity、配对初始化、G屏蔽列零梯度、独立double全部4770参数梯度及固定1e-4/1e-5差分通过；最大梯度误差5.551e-17。
- G/H各真实513训练窗+12验证窗、两epoch34更新、19步中断序列化恢复，全部模型／optimizer／TorchCPU-CUDA／loader／Python／NumPy RNG、批序和history EXACT。坏source/stat/split/output实际拒绝。不是全3103两轮，不加载旧F16/decoder/scorer，不是外部评审。

## 端点tiny为什么没有运行

原文为“64个含非identity教师动作的窗口+64个其余窗口”，现有冻结核心代码将第二层实现为不含非identity教师动作的窗口。真实END训练集3103窗全部属于第一层，第二层为0。候选级仍有11973个identity标签，不等于存在非活跃窗口。

因此登记STOP_TINY_STRATUM_INSUFFICIENT，tiny更新0，90%下降测试未执行；没有给出假的tiny拟合阴性或完整预检PASS。TIME已独立在P0停止，不能因为它有750个非活跃train窗就跨分支借样本。

建议仅供确认的最小澄清：第二组改为从第一组未抽中的train窗口继续按同一固定哈希取64个，允许仍为活跃窗口；保持128个不同train窗、256次更新、原90%门及全部正式学习合同不变。此澄清尚未执行，未换LR、宽度、标签、GT或评价阈值。现有P0与GPU恢复证据可保留，确认后按新source绑定补tiny及剩余保存/benchmark预检。

## 当前未执行

正式G/H六模型、完整8epoch效果、hold/inner GT本轮评价、六比较主CI、真实模块成本和P3联合均未执行；P3还因TIME的P0门失败而不触发。不得声明部署成功、major提升、独立whole-model泛化或升级baseline。没有新loss/scorer/unfreeze/data/额外变体，旧PAUSED和所有文件保留。

## 同助手反方审核DA2/DA3（非外审）

- DA2：134个安全oracle救回并不证明可学。GT选出的最优参数可能多解；H/G只检验显式历史增量，G输入本身已有冻结历史信息。TIME的50→24→5是按顺序的过滤统计，不是互斥的因果根因分解。
- DA2：窗口级无非活跃层是抽样合同可实现性问题，不能偷换为模型表达、优化或观测信息失败。相反也不能因安全oracle有机会就跳过tiny或改变样本层使预检自动通过。
- DA3：所有安全oracle通过、无lost的结论只对当前GT特权有限动作和人口成立。未训练/未held对比即不支持主要研究问题。原重大改善、原成功保护、全部motion和相关性标准不变；本轮报告明确保留这种未完成性。

## 完整证据

- p0_completion_review/REVIEW.json、ALL_GROUPS.json/csv、各oracle完整NPZ。
- completion_review/P0_WINDOW_EVENTS.csv：24576个重复port-window事件与ID，含全部rescue/lost/retained/unresolved。
- completion_review/P0_AMBIGUITY.csv、FULL_MOTION_TABLE.csv：所有组分母、多解及完整运动量比值。
- END_learning_preflight/GPU8.json、RESUME.json、READER_REJECTION.json、TINY.json及RESOURCE.json。
- TINY_STRATUM_ADMISSION.json、learning_source_extension.json：真实分层及29个原partial来源核验。

报告由AI助手辅助生成与审核；源码、原始数组及SHA收据是证据，不是外部独立评审或第二次全模型梯度回放。
