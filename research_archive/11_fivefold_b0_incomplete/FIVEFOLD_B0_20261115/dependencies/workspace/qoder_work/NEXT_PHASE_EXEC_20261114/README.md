# NEXT_PHASE 实际执行目录

用户于 2026-11-14 授权执行 Downloads/1/NEXT_PHASE_DESIGN_20261114。原始文件逐字保存在 design_original/；所有训练和新增证据只写入本目录。

执行顺序：E1-A 七次新训练及已有 B0 seed42 → E1-B FIT85 八次独立预算训练 → 按 CALIB 规则选 E* → 必要时 E1-C 五个新 B0 → E2E_V2_CONTRACT.json 与 SHA 封存 → 方法真实 GPU 预检 → E2 零训练诊断 → E2/E3/E4 共 27 次单因素训练 → 全模型 final 锁定 → fit/dev 导出及完整双实现质量/聚簇/运动审核 → 整 track+seed 统计和报告。

T3 已经用户确认改为旋转逐步位移再累加，保证每步速度大小不变。新增模块构造保护 CPU 和 CUDA RNG。E3 堆叠零初始化输出层的梯度分阶段检查延伸至第4步；不改变科学结构。

运行状态见 CURRENT.json、WORKER_START.json；完成必须同时出现 RESULT.json status=SUCCESS 和 WORKER_EXIT.json exit_code=0。FAILURE.json 保留失败现场，不自动重试。每次独立训练的 logs/ 与 receipts/ 记录实际子进程退出状态；不能把启动或预检 PASS 当作科学成功。

训练预算最多 E*=64，预计需要数小时，实际时间由 PREFLIGHT_E1.json 和各 epoch 实测给出。worker 是无周期触发的按需 Windows 任务；退出对话后继续运行，需电脑保持唤醒并维持登录。没有注册周期监控或自动扩展研究。

最终结果见 RESULT_ANALYSIS.md / .json、analysis/；所有旧门另列 LEGACY_SHADOW，旧结论和 baseline 不修改。C0 是反复使用的开发划分，不是独立盲测。5-seed 噪声地板和设计的阈值仅为探索性筛选，不包装成正式显著性。E2 零训练结果是诊断，不能保证是训练效果的数学下界。F 阶段组合及 ETH/UCY 五折不在本次执行范围。
