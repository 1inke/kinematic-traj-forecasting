# 资源规则修改记录

2026-10-15人类明确批准按剩余任务需要内存×1.3，而非固定6GiB判断。

已完成：
- 新resource_policy.json与memory_policy.py替换固定系统余量和固定process上限；逐任务估计+1.3，运行中扣除已分配RSS/private。
- common.py恢复适配器实际重定向legacy.guard与聚簇算术ns.guard，父common.py/科学文件/失败证据字节未改。
- 只读NPZ头估计及边界、已占用扣除、独立physical/commit、估计超限和坏估计测试PASS。
- 估计本审核总峰值4.283550262GiB（历史实际峰值高于静态payload估计），启动完整阈值5.568615341GiB。
- 仅一次当前资源快照2026-10-15T20:31:49.537859HK：physical_available855969792bytes（约0.7972GiB），commit14934802432bytes（约13.9091GiB）；启动/运行中判断均因物理内存不足STOP。
- 对旧故障瞬间数据回放新runtime规则可PASS，说明已占用扣除与固定6GiB门不同；这不代表当前0.80GiB可启动。

未进行：审核启动、worker/task创建、训练重启、模型前向/optimizer、新科学结果分析、OS/pagefile或用户程序调整。旧1916源清单未覆盖、旧失败/结果未删。
POLICY_VALIDATION只是资源配置通过，不是科学审核通过；当前仍等人类释放内存并明确继续。之后只恢复保存审核，不重训六次。
