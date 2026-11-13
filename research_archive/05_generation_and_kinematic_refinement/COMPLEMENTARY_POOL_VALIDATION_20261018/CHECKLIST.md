# 执行清单（对照 NEXT_VALIDATION_DESIGN）

日期：2026-10-18。主合同：GROK_KINEMATIC_REVIEW_20261018/NEXT_VALIDATION_DESIGN.md

## V0 — 仅补证据与保存算术，不训练
- [x] 输入冻结 manifest
- [x] 查找原 HGB / 40scores / selected_indices / selected_paths
- [x] 若缺失：写 `ORIGINAL_SELECTOR_REPLAY_UNAVAILABLE`
- [x] 不静默重训 generator

## V1 — 重建并冻结选择器（生成器不动）
- [x] 固定主方法 `cond_boost_11_9`；不按新结果改主方法
- [x] HGB 超参锁定；二元 good 标签；声明 soft 退化为 good
- [x] FIT 内按完整 target track 划出固定校准部分
- [x] 稳定并列：分数降序，并列按物理 slot 升序
- [x] 每 split/seed 保存完整产物（scores/indices/paths/HGB）
- [x] 参照 native20 / KIN20 / union40 / cond_top_11_9
- [x] 每 group：coverage/rescue/lost、minADE/FDE、J、五 motion、聚簇、五类窗口、delta、lost 归因
- [x] track×seed 层级 CI（family8）；标注探索性

## V2 — 证据闭环后
- [x] 条件 HGB vs geometry-only

## 自改进（合同完成后）
- [x] kinsafer / natunique（FIT_cal 选）
- [x] 固定配额 11–14（保留原生支撑）
- [x] 自适应原生配额
- [x] 小锁定消融，不扩模型、不大扫参
