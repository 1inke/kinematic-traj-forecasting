# 本轮复核范围与读取器修正

仅 CPU 重算现有 native/KIN 缓存与已保存 K20 索引，无模型加载、预测、训练、求解器或后台任务操作。旧文件不改。

首次读取器在 seed42 outer/inner 重算后，遇 original_CAL 精简 JSON 缺少 `rescue` 键而退出（KeyError）。这是本轮复核读取器 schema 假设错误，不是原实验失败；未写完成复核结果。修正后对实际存在的字段比对，CAL 缺失的 rescue/lost 仍从原路径重新计算；其运动指标比对已保存 ratios，其他 split 比对绝对值。阈值未改。

不把本轮读取前后 SHA 一致当作原训练时 source-lock，不复算聚簇和 bootstrap，不加载 HGB 或重放所有模型。

第二次在 CAL V1 的索引 array SHA 比较处退出。核对 `run_cal_eval.py` 后确认：文件保存 int16，但原评估明确转 int64 再计算 array SHA；本轮最初直接按存储 dtype 计算。按原评估口径转 int64 后匹配，文件原字节仍单独 SHA 保护，不改文件、不忽略不匹配。
