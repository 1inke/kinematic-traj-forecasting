# 资源检查时点适配：科学完全等价的新执行版本

当前human授权完成同一次K20独立/协同细化验证、定位实现错误自行处理，不删除任何东西。父K20_COLLABORATIVE_REFINEMENT_EXEC_20261027 worker34104在2026-10-27 17:01:47.226676双FAILURE，第一子train_B1_42在创建run/模型/optimizer前STOP_RESOURCE，0正式更新，全部旧源码/日志/任务/源锁3930和检查保留，不重开父或重训完成模型。

父启动门在train.py先Data()及Torch导入之后调用startup，已实际分配TASK private1060585472/RSS663769088，又预留完整E4294967296×1.3=5583457484.8，系统availablecommit4582178816，因而拒绝。父policy的1.3 remaining本应在该时刻应用：commit余需1.3*(4294967296−1060585472)=4204696371.2，此次原读数本身可满足runtime；不把该推算伪称当时完整startup实测PASS。

资源适配v1保持原E时，真正pre-Popen检查读数physical13342576640/commit5247987712，提交余量仍低于5583457484.8，已真实STOP、0训练，失败完整保留。**以下v2为依据实际本任务预检峰值的显式估计修订，替代v1“E不下调”的计划，不把v1称成功。**

按human既定“估算任务实际需内存再乘1.3”，从同科学代码已完成GPU8/fullbatch32/36benchmark/513恢复/saved128真实Windows峰3144228864，加最大实际输入NPZ未压缩载荷115885056，再加32MiB检查点快照与256MiB分配器固定余量，得E3562103808。公式不使用当前available反推，不改1.3或任何科学门；MEASURED_MEMORY_ESTIMATE逐项来源。原4GiB工程储备约1.0GiB不是最低需要的实测值，原失败不洗掉。新current/lifetime峰>1.3E照样STOP重估。

协调进程在Popen前实际核bothavailable>=1.3E并保存每阶段admission；子进程导入/数据分配后按runtime remaining和current/Windows lifetime peaks检查。预检也由独立无Torch协调器先startup后GPU/CPU子进程，证明真正执行路径。纯CPU锁模型元数据阶段从安全侧仍按GPU E预留。准备数据经新prepared junction只读复用父相同文件和SHA，不复制/重建统计；不调用prepare写原数据。

模型、loss、插值、优化器、train批顺序、所有preflight数学和两eps、评估/聚簇/五motion/bootstrap/cost协议逐字节复制；只core运行命名空间/源继承与resource时点、实测估计、worker child前admission和任务名字变化。SOURCE_EQUIVALENCE逐文件/AST证明，科学protocolJSON SHA完全一致，resource公式/factor/CPU不变、GPU estimate明确修订。整个父执行目录完成产物继承为只读源，新锁后不再编辑。完整当前GPU/CPU预检+真实admission+ETA后新soleOS0trigger；实际1.3准入仍不足则停止，不盲重试或改科学标准。

