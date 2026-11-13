# 资源检查时点适配：科学完全等价的新执行版本

## 当前资源v3（最新human明确要求，覆盖下文v1/v2历史策略）

2026-10-27人类指出检查机制严重影响正常使用，明确要求修改。下文v1/v2失败和策略保留为历史，不再当当前门槛。当前新源码尚无生产锁/worker/正式更新，允许资源基础设施显式修订；旧父科学源码与3930源锁不改。

新resource_guard_v3将1.3乘预估剩余需求改为预警，而不是不分情况立即中断。轻量协调器在子进程分配前核新阶段准入：available physical和GetPerformanceInfo真实system commit各能容纳E加系统余量（物理总量1%，界128–512MiB）；约32GiB机器余量约324MiB。不重复预留已分配内存；运行时剩余量分别扣当前RSS/private。E不反推当前可用量、仍3562103808，未静默抬高或降低。

超过1.3E/历史生命周期峰值只留预警和估计偏差证据，不再自动杀死健康任务。运行中只有当前可用物理或提交余量低于自适应系统余量连续至少10秒且至少3次实际检查，或小于64MiB紧急线，才在原安全检查点拒绝继续。每2秒内复用读数，阶段最终报告强制实测；不添加常驻轮询或等待，不设置OS/pagefile，不关闭用户进程，不锁定/预占内存。此守卫降低错误拦截，不保证替代操作系统OOM或捕捉每次瞬时分配。

RESOURCE_POLICY_TESTS实际注入旧假阳性、已分配扣除、历史峰超过、短时压力恢复、持续真实压力、紧急余量和读数失效；随后全部原GPU/CPU科学预检及真正pre-Popen流程重跑。v2代码/报告全文在resource_policy_v2_snapshot保留，原候选manifest由candidate_provenance保留。scientific protocol与模型/梯度/数据/阈值完全不变；SOURCE_EQUIVALENCE明确resource公式已获human授权改变，不冒称旧公式EXACT。

## 以下为原v1/v2背景（非当前门）

当前human授权完成同一次K20独立/协同细化验证、定位实现错误自行处理，不删除任何东西。父K20_COLLABORATIVE_REFINEMENT_EXEC_20261027 worker34104在2026-10-27 17:01:47.226676双FAILURE，第一子train_B1_42在创建run/模型/optimizer前STOP_RESOURCE，0正式更新，全部旧源码/日志/任务/源锁3930和检查保留，不重开父或重训完成模型。

父启动门在train.py先Data()及Torch导入之后调用startup，已实际分配TASK private1060585472/RSS663769088，又预留完整E4294967296×1.3=5583457484.8，系统availablecommit4582178816，因而拒绝。父policy的1.3 remaining本应在该时刻应用：commit余需1.3*(4294967296−1060585472)=4204696371.2，此次原读数本身可满足runtime；不把该推算伪称当时完整startup实测PASS。

资源适配v1保持原E时，真正pre-Popen检查读数physical13342576640/commit5247987712，提交余量仍低于5583457484.8，已真实STOP、0训练，失败完整保留。**以下v2为依据实际本任务预检峰值的显式估计修订，替代v1“E不下调”的计划，不把v1称成功。**

按human既定“估算任务实际需内存再乘1.3”，从同科学代码已完成GPU8/fullbatch32/36benchmark/513恢复/saved128真实Windows峰3144228864，加最大实际输入NPZ未压缩载荷115885056，再加32MiB检查点快照与256MiB分配器固定余量，得E3562103808。公式不使用当前available反推，不改1.3或任何科学门；MEASURED_MEMORY_ESTIMATE逐项来源。原4GiB工程储备约1.0GiB不是最低需要的实测值，原失败不洗掉。新current/lifetime峰>1.3E照样STOP重估。

协调进程在Popen前实际核bothavailable>=1.3E并保存每阶段admission；子进程导入/数据分配后按runtime remaining和current/Windows lifetime peaks检查。预检也由独立无Torch协调器先startup后GPU/CPU子进程，证明真正执行路径。纯CPU锁模型元数据阶段从安全侧仍按GPU E预留。准备数据经新prepared junction只读复用父相同文件和SHA，不复制/重建统计；不调用prepare写原数据。

模型、loss、插值、优化器、train批顺序、所有preflight数学和两eps、评估/聚簇/五motion/bootstrap/cost协议逐字节复制；只core运行命名空间/源继承与resource时点、实测估计、worker child前admission和任务名字变化。SOURCE_EQUIVALENCE逐文件/AST证明，科学protocolJSON SHA完全一致，resource公式/factor/CPU不变、GPU estimate明确修订。整个父执行目录完成产物继承为只读源，新锁后不再编辑。完整当前GPU/CPU预检+真实admission+ETA后新soleOS0trigger；实际1.3准入仍不足则停止，不盲重试或改科学标准。
