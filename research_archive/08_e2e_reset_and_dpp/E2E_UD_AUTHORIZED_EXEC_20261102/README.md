# U/D 四个单因素 pilot：获批修订执行版

本目录承接用户明确同意的 ARS 窄修订；原设计、B0、上一轮 STOP 诊断和失败证据均保持不变。当前状态以 runtime_control.json、WORKER_STATE.json 和 WORKER_EXIT.json 为准，不按 README 快照重启。

## 授权与不变量

- 实际 B0 context/mode **64维**、clip **5**，不采用原稿误写的96维/clip1。
- 三阶段真实全模型梯度门：初始新增残差零且路径/打分等价；原头首步先打开；第2步输出投影梯度正；第3步全部内部张量梯度正。不将已证实的首步零梯度写成 PASS 或永久死锁。
- U-GRU、U-Delta、D-Gate、D-DiffGate 各 fresh seed42 固定8轮，2720更新，共32轮10880科学更新。原 B0 初值、所有批次与增强随机流配对；新模块构造使用隔离 RNG。新旧全部活跃模型参数训练，非冻结 B0 主干或已训练权重 warmstart。
- C0 fit21719 /dev4753、past-only 因果邻居、原目标与 AdamW/EMA/scheduler/augmentation 不变；最终第8轮 EMA，不按 dev 挑 checkpoint。不叠 U+D、不新 loss/scorer/数据。
- 参数 U-GRU301618、U-Delta302514、D-Gate307986、D-DiffGate312082；D 两臂明确超过305000目标，不砍结构凑数。整模 raw-observation CPU/GPU 成本另有实际收据。

## ARS DA1 — 合同修订审查（同助手角色，不是外审）

Critical 旧冲突已由实际源码/梯度证据及用户授权解决：64/clip5/三阶段门；D超目标公开披露。不声称授权能把科学阴性改成阳性。

Major 保留：额外容量、优化几何与特定递推/差分机制不能混为因果；必须同时看 GRU/Gate 对照。原稿把零初始化旧头首步梯度0解释成永久锁死的说法不成立。差分权重可带负值且行和.5，不是路径概率/心理意图。

Minor 保留：固定前64个 FIT 窗的成本测量不是所有邻居密度的上界；ATen attention/mm/bmm FLOP 不是全部非线性/NumPy scalar 指令数，后者包含在延迟中。

单seed不报告确认性显著性。fit-past 75%阈值、dev固定hash128的末帧±.05m扰动只描述机制；未知姐妹录像独立性和反复开发局限保留。零参考尾部 p95 不加 floor/epsilon，不从0/0推出相对非劣。

## 管线与后续

preflight_ud.py：四臂 GPU、NumPy/functional 梯度与固定两 eps FD、513窗两轮精确恢复、128保存全质量/原聚簇/pdist/运动双审核。prelaunch_supplement.py：四臂各实际 FIT 第一轮 RNG/order 对 B0 精确一致，晚加入的机制/配对消费者真实分支验证。这些夹具更新不是科学结果。

worker.py：按需 Windows XML0任务，仅一个 coordinator，串行四个独立 train_arm.py 子进程，退出后 compare_ud.py 做保存审核/比较/ARS DA2和DA3报告。生产前 source_manifest 锁源；当前源码锁后不可静默编辑。失败保留，不重复已完模型、不删除文件或改旧 PAUSED。

先完成本四臂；后续 seed43/44 仍须按原门及各自预检/预算准入。主候选固定 U-Delta /D-DiffGate，不把更好的对照重命名为主方法，不通过主观“机制线索”自动越门。不在此 worker 自动启动 V/F/多日训练。

资源沿最新 warning-first v3：实测+明列 inventory 的 E；1.3只是预警；真实低余量持续/紧急才安全停止。无固定6GB、锁内存、OS/pagefile修改或用户进程 kill。所有文件和旧结果永久保留。
