# 当前参考与 baseline 证据

截至2026-10-20，本轮未产生或晋升新的 S4 baseline。历史 baseline 保持原有定义；不能将“尚无按新协议完成的 S4”解释为旧实验被撤销。

新框架的冻结比较对象是原 F16 **native20** 缓存：8步已准备观察、12步未来、dt=.4s、固定20槽原顺序。路径来自冻结缓存 `candidate[..., :24]`，不是本轮重训或重新运行 F16。P2 是冻结编码器 observed tokens 的旧导出；不能因缓存冻结就宣称编码器从未见 outer。

冻结文件及实际 SHA 见 [config/data_manifest.v1.json](config/data_manifest.v1.json)。newhead 的25,304参数不等于整个 native 模型参数量；新 G5 的305k预算必须核算实际整体模型，当前 A/B adapter 尚未完成该核算。

工程回放中 native outer 为625/1035，outer change 为110/253（68 tracks）；这是历史缓存参考，不是新训练三 seed 的独立重复。详见 [D-0 RESULT](experiments/2026-10-20_D0_protocol_calibration/RESULT.json)。

来源限制：native h3/v2 的实际训练 split 文件包含 outer 身份及时间区间；original_CAL 又与当前 FIT 存在直接时间交叠。见 [重叠审计](TRACK_OVERLAP_AUDIT_20261020/RESULT_ANALYSIS.md)。新 S3 必须先关闭处置项；S4 解封、标准五折、5seed 与 baseline 变更仍须单独人工签字。
