# 结果报告（short 三 seed：42/43/44）

时间：2026-10-18T13:07:11+0900（Asia/Tokyo）

## FACTS（事实）

协议：8 epoch，全 FIT，batch 256，K=20；评价 outer overall；质量 ADE<=0.2 且 FDE<=0.4；motion_factor=1.01。

### 每 seed outer overall

| seed | arm | coverage | minADE | minFDE | motion | fails | accel_ratio | speed_p95_ratio | weak_positive |
|---:|---|---:|---:|---:|---|---|---:|---:|---|
| 42 | BASE | 0.4271 | 0.3143 | 0.6245 | FAIL | speed_p95,accel_mean,jerk_mean | 1.801 | 1.028 | - |
| 42 | KIN | 0.5952 | 0.1962 | 0.3513 | FAIL | speed_p95 | 0.963 | 1.070 | False |
| 43 | BASE | 0.4058 | 0.3316 | 0.6500 | FAIL | speed_p95,accel_mean | 1.371 | 1.168 | - |
| 43 | KIN | 0.5536 | 0.1924 | 0.3367 | FAIL | accel_mean | 1.060 | 1.004 | False |
| 44 | BASE | 0.4164 | 0.3242 | 0.6352 | FAIL | speed_p95,accel_mean | 1.732 | 1.208 | - |
| 44 | KIN | 0.5585 | 0.1924 | 0.3484 | FAIL | accel_mean | 1.031 | 1.008 | False |

native coverage（三 seed 相同分母）= 0.6039

### 三 seed 均值 ± 样本标准差

| arm | coverage | minADE | minFDE | accel_ratio | speed_p95_ratio | motion PASS 数 |
|---|---|---|---|---|---|---:|
| BASE | 0.4164±0.0106 | 0.3233±0.0087 | 0.6366±0.0128 | 1.635±0.231 | 1.135±0.095 | 0/3 |
| KIN | 0.5691±0.0227 | 0.1937±0.0022 | 0.3455±0.0077 | 1.018±0.050 | 1.028±0.037 | 0/3 |

KIN−BASE coverage：均值 **15.27 pp** （std 1.37 pp）
KIN/BASE minADE：均值 **0.599** （std 0.023）
预注册 weak_positive 通过数：**0/3**
全部 seed cov_ok=True；全部 seed ade_ok=True

### 产物路径
- seed 42: <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\COMPARE_seed42.json
  - <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\BASE_seed42_RESULT.json
  - <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\KIN_seed42_RESULT.json
- seed 43: <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\COMPARE_seed43.json
  - <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\BASE_seed43_RESULT.json
  - <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\KIN_seed43_RESULT.json
- seed 44: <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\COMPARE_seed44.json
  - <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\BASE_seed44_RESULT.json
  - <WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\short\KIN_seed44_RESULT.json
- 汇总：<WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018\results\SUMMARY.json

## INFERENCE（解读，非门控改写）

三 seed 上 KIN 相对 BASE 的 coverage 与 minADE 改善方向一致；严格运动学门仍未过（各 seed KIN 均 FAIL，失败项在 speed_p95 或 accel_mean 上略超 1.01）。accel/jerk 相对 BASE 明显更接近或低于 native。这支持结构偏置有助于平滑与误差，但不构成预注册弱阳性通过，也不能外推模式保持或 P1。

后续：可选：新设计号速度软限幅专治 speed_p95；或停止本结构版本。不扫一致性权重/隐层；不接 P1/P2。

未改 MODE_PRESERVING/P1；未做 λ/隐层扫参；未跑速度软限幅消融（留待新设计号）。
