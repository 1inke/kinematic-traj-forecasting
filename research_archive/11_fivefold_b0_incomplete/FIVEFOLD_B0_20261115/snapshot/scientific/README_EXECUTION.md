# 五折 B0 独立执行

本目录执行用户提供的 FIVEFOLD_B0_20261115。原设计和八份原代码完整保存在 design_original；code 中对应八文件不改。新增 runtime.py 仅负责来源校验、单线程策略、原子保存、完整轮次断点与运行状态；实际模型、Trainer.step、数据增强、损失、优化器、EMA、指标、预算规则不变。

## 已确认的数据协议修订

原 F0/AUDIT.json 保留 univ/zara01 的 A3_scene_overlap 失败。用户明确选择保留原数组，以录制序列隔离取代粗 scene ID 不交叉。F0/AUDIT_RECORDING.json 将全部 158671 个目标 20 帧路径逐项匹配回原录制标签；每个窗口唯一来源，零未匹配、零歧义、零反侧录制路径匹配。共享 homography 编号仍可能代表同一物理位置，不能称为物理地点完全隔离。其他审核条件不变。

F0 固定完整性审核会读取测试数组；预算模型不使用测试指标。只有 10 个预算运行完成并校验、固定 E* 决定封存后才允许模型测试评价。DECISION_SEAL 明确记录这一区分。

## 运行与恢复

双击 standalone/START.cmd：启动或手动继续同一有限实验。双击 standalone/STATUS.cmd：只读查看最新保存进度。Windows 按需任务 Trajectory-Fivefold-B0-Standalone-20261115 无定时触发器、无限执行时限、IgnoreNew 和命名 mutex 防重；不依赖 GPT/Codex/API 或网络。

退出 Codex 后仍运行。运行期间临时阻止系统自动睡眠，屏幕可关闭；需要电脑开机并保持登录。主动关机、重启、注销或手动睡眠会中断；重新开机登录后手动 START 从最近完整 epoch 恢复。中断当轮会重算。每个 run 仅保留最近两个权重断点及完整恢复状态，旧实验文件不删。不自动失败重试或改变科学设计。

F1：5 折 × 64/128 轮 × seed42，共 10 次。F2：五折平均相对 ADE 改善 >=1% 且平均 FDE 不变差时选 128，否则 64；>=3% 只标可能未饱和，不扩展 256。F3：5 折 × seeds42/43/44，共 15 次完整训练，固定最终 EMA。F4 自动写 FIVEFOLD_RUNS/SUMMARY.json 和 SUMMARY.md，再核验并封存全部结果。

保存 val/test 完整 float32 预测、logits、窗口 ID/scene/group 与逐窗指标，以备后续分析。DONE 的输出 SHA 校验通过后跳过完成运行；不重新训练。真实 GPU 预检包含 513 窗 2 轮的所有状态精确恢复、未来标签扰动不改变推理，以及重复保存/已完成跳过核验。预检与正式 FIVEFOLD_RUNS 分离。

EXP156/133 为不同选 checkpoint 协议的单 seed 量级参照。当前差异不能独立证明组件贡献或预算作用；文献协议尚未核验。本次不升级 baseline，不增加方法，不自动推送 GitHub。

当前状态以 standalone/CURRENT.json、CHILD_STATE.json、WORKER_EXIT.json 和实际任务/进程为准。历史 preflight 或 prior 文件不是当前训练进度。SOURCE_MANIFEST 固定输入/代码，最终成功须 RESULT SUCCESS、WORKER_EXIT exit_code0 与 FINAL_OUTPUT_SEAL 同时通过。
