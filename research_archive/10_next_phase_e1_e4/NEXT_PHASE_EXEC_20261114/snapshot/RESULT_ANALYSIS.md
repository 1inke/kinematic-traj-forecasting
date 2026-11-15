# 下一阶段实验结果

E*=64；E1→合同封存→E2/E3/E4 已完成。

重复使用的开发集，仅为探索性研究；噪声标尺不能当作正式显著性检验。五折和因素组合未执行，baseline 未升级。

| 方法 | change覆盖差(pp) | E2E_V2伤害组 | 预定规则 |
|---|---:|---|---|
|E2_natural|1.2153|overall,stable,transition,change|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|
|E2_clamped|0.9722|overall,stable,transition,change|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|
|E2_clamped_launch|0.7292|overall,stable,transition,change|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|
|E3_self|3.0556|stable|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|
|E3_set|3.9931|overall,stable,transition,change|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|
|E3_set_nobias|4.4444|overall,stable,transition|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|
|T1|4.9306|overall,stable,transition,change|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|
|T2|1.2153|overall,stable,transition|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|
|T3|0.3472|无|NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN|

完整 fit/dev、逐 seed、救回/丢失、GT参照运动、尾部、LEGACY_SHADOW、整track+seed区间见 RESULT_ANALYSIS.json 和 analysis/。

T3 使用用户确认的逐步位移旋转方案。A7 是否复现历史 B0 及其库版本局限见 E1/E1A_NULL.json。