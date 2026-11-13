# 同助手反方检查：T2 与 A/B/C 有限系列

## DA2：先保留能反驳阴性叙事的证据

- C 不是没有学习：FIT change 的三 seed 覆盖均高于 native；相对 A 多 10/27/3 个窗口。外层总体覆盖 638/634/640 也均超过 native625；不能把 change 阴性写成所有群体都失败。
- C 相对 A 的 fixed-native-j0 实际 ΔJ 三 seed 点估计均改善（−.00925235/−.00849398/−.00015341），CAL change 也由 A221/214/225 变为 C223/230/229。说明局部表示改变有可观察效果，不能称“完全无效”。
- 但 primary 是 outer change：C108/111/109 对 A107/105/113，第三 seed 下降；C−A coverage family24 CI [−4.781,+5.191]pp 跨零，且均值仅 +.395pp。关键slot family24 CI 同样跨零；这些正例不足以通过迁移联合门。
- B 的 FIT 保留明显改善、CAL比A好，必须保留；同时 FIT rescue减少，不能把少lost自动解释为高质量生成成功。T1旧资源偏差仍是独立执行限制。
- 宽 CI 是不确定，不是等价/不可能/无信息；不追加seed或换family凑显著，不把三个seed复制native视为三份独立对照。

## DA3：报告与停止决定的反证检查

- C 的外层 change fixed-j0 实际 ΔJ 仍 +.030784/+.026719/+.039120（越大越差）；ALL20平均 ΔJ 却负，不能用ALL20平均改善替代最关键原始slot、samecandidate完整覆盖或归因所有方向错误。
- C outer change 的 lost 为17/15/19，rescue15/16/18。净覆盖−2/+1/−1掩盖了约13.64–17.27%的原生成功丢失；并非保留门通过。
- C 全48个seed×split×group仅accel_mean超1%，其余四pooled运动学项通过。必须明确是哪一项坏，不能泛称速度/jerk全面崩溃；也不能用p95通过抵消平均加速度失败。
- 执行PASS与科学signal分开：3个C模型、75history/5550step、28job与独立保存审核通过；T1旧resource guard漏生命周期峰值的偏差并未因T2修复而消失。
- 2560源+已完成输出前后SHA和CPU补充144组算术max0通过；补充没有Torch、模型前向、训练或全梯度回放。生产独立审核已复算聚簇和20k bootstrap，本补充不伪称再次全部重算。
- T3必须两个机制门同时成立；T1 false、T2 false，因此D NOT_TRIGGERED。不能以人类实现错误自修许可扩大到新loss/MoE/scorer/旧模块解冻/新数据。
- 所有结论限于F16已见、3scene/3seed、target非context-disjoint且outer/CAL适应复用的探索。不是独立wholemodel泛化、SOTA、心理意图或唯一shared-head根因。

结论：保留正例，仍停止这个有限A/B/C系列；不升级baseline，不启动D，不再做本系列局部扫描。

说明：依据 Academic Research Suite 的证据—反证流程，由同一助手依次完成，非独立外审、非多代理盲审。
