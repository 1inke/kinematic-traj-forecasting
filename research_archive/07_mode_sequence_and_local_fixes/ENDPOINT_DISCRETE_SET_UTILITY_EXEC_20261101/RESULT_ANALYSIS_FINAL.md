# 有限端点动作集合效用：最终实验报告

结论：`TINY_ACTION_UTILITY_STOP`。正式F阶段：`NOT_TRIGGERED_T_GATE_FAILED`。

## 固定方法与范围

原tiny128窗/97相关tracks，Native69成功；新6800参数G/H各3fresh种子，48→64→32→48。原40维统计冻结，新8维集合距离仅train3103拟合；G在标准化后清零历史16维。原END49去掉重复identity后20×48单槽编辑+全局NOOP，原guard不变，其他19路径逐字节保留。

监督是所有合法动作的完整K20效用，正/负/零非空类别等权MSE，float64比较；没有唯一oracle教师、连续C、量化Q、额外保护loss或GT部署回退。NOOP阈值固定0，分数不是校准概率。

## 六个固定末步结果

|arm|seed|覆盖/128|救回|丢失/69|NOOP|负效用选择|有限动作regret|
|---|---|---|---|---|---|---|---|
|G|42|72|5|2|0.000|6|0.246367|
|H|42|76|7|0|0.000|2|0.201065|
|G|43|72|4|1|0.000|5|0.241293|
|H|43|74|6|1|0.000|5|0.228483|
|G|44|72|5|2|0.000|7|0.249206|
|H|44|76|7|0|0.000|3|0.210409|

H为事先指定主方法；不能改用最好G推进。H每seed必须lost0、rescue≥9/9/10并通过全部protected误差、change不坏、全组五pooled及junction门。

- H42：{'zero_lost': True, 'rescue_min': False, 'nonidentity_rescue': True, 'protected_error': False, 'all_motion_junction': True}；联合通过=False。
- H43：{'zero_lost': False, 'rescue_min': False, 'nonidentity_rescue': True, 'protected_error': False, 'all_motion_junction': True}；联合通过=False。
- H44：{'zero_lost': True, 'rescue_min': False, 'nonidentity_rescue': True, 'protected_error': False, 'all_motion_junction': True}；联合通过=False。

所有seed×保存点0/64/128/192/256×四组见ALL_GROUPS.csv；全部3840窗口事件及选择分数/真实效用/救回丢失见WINDOW_EVENTS.csv，最终每窗运动成本另见completion_extension/FINAL_WINDOW_EVENTS_AND_MOTION.csv。不是只报净增或最优中途模型。

## D0机会与审计

旧O0路径逐字节复现21rescue/0lost；新效用oracle另外计算，同样覆盖90且实际全组误差/运动通过。两者皆GT见证，不是可学习或可部署证书。

D0逐动作显式重构122880个K20集合，标签/符号/同路径覆盖与公式完全一致，maxerror=0.0；新8维train-only独立两遍与31/256chunk统计最大差6.04e-14。原train/val IDs和sequence|target tracks无交集；不是上下文独立或从未开发过的数据证明。

实际G/H GPU8 phased、独立NumPy全6800梯度/固定双eps1e-4/1e-5、配对初值/归一化后历史清零梯度0、NOOP/ties/全invalid分母/GT改动前向mask选择不变均通过。双方真实hash513train+12val两epoch34updates、第19中断，参数/Adam/externalstats/全部CPU-CUDA-Python-NumPy RNG/sampler/batch/history/trace EXACT。不是全train两轮。

两臂实际128非零输出在科学启动前完成独立CPU同路径质量、原SciPy完整K20 complete-link(.10/.20)+pdist、每组pooled五motion及junction、19固定槽、标签/选动作/regret保存审核。坏source/split/stat/actionorder/checkpoint/output真实reader拒绝。

完整结案审核6checkpoint/1536次preupdate损失/30保存点，算术最大loss差5.55e-17；5853来源初末SHA不变。保存算术不是所有模型梯度二次前向后向，同助手ARS DA1/2/3非外审。

## 代价、边界与结案

保守T启动规划288.72s，门1800s；实跑训练保存61.94s，独立审核45.95s，另有D0/预检/封存。科学1536 Adam更新，预检208（恢复136+固定benchmark72），总实际1744；不混计。无失败前科学更新、无旧模型重训。

实际8窗口集合距离0.001209s，原960 guard0.127919s；6800参数/133120 MAC只指新MLP，不含原F16/数据IO等真实全管线成本，不称整模轻量部署成功。

resource_guard_v3为实测inventory估计，1.3仅预警；各受测进程自身Windows生命周期private/RSS峰见EXECUTION_ACCOUNTING_FINAL.json，不是全树合并峰。无固定6GiB硬停、无OS/pagefile/用户程序kill或内存锁。原独立按需XML0任务保留记录、不重复startup/half或恢复旧PAUSED，所有文件/模型/日志/失败都保留。

T联合门失败，按设计停止此固定端点小分支；F六模型/H1-H5五项held检验未触发，hold/inner GT未解析或评价。未改阈值/系数/seed/epoch，也没有挑G替代H。

新方法相对旧RS改变表示、目标和单槽政策，不作单因素因果归因。G已有F16历史处理后的native几何，H-G只检验额外显式历史且此处是训练内tiny；6800容量、8维摘要、类别重加权/960最大选择误估和有限优化均为竞争解释。不能由当前结果推导历史无信息/未来不可辨识，也不能把tiny改善当跨track泛化或心理意图理解。旧90%STOP/TIME/P3、major双5%及全部1%保护/相关性审核、baseline/historycycle11与长期目标保持。
