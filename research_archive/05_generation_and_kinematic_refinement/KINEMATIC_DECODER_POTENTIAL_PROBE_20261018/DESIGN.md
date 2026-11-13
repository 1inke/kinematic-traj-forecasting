# 设计：KIN+clamp 潜力探测（冲 native 覆盖）

## 问题
SPEED_CLAMP 短训（8 epoch）已达 weak_positive 3/3、motion PASS 3/3，但 outer 覆盖均值 ~0.565 < native ~0.604（约 −4pp）。问：在**更长协议**下，同一 KIN+软限幅配方能否追上/超过 native 覆盖，并保持 motion PASS 与 ADE 优势？迁移如何（FIT→outer，兼看 CAL）？

## 假设
短训可能欠拟合；加长到 32 epoch，同一限幅下覆盖可能继续抬升。若仍平台低于 native，再试**一次**轻度放松限幅（仍软限幅，非扫网格），看天花板是否被限幅卡住。

## 臂 / 变体（最多 2）
| 变体 | 说明 |
|---|---|
| **LONG** | 与 SPEED_CLAMP r1 相同限幅：`v_max=native_speed_p95×0.995`，`a_max=native_accel_p95×0.90`，`motion_w=0.05`；**32 epoch**，3 seed |
| **LONG_RELAX** | 同上但限幅略松：`v×1.00`，`a×0.95`；32 epoch，3 seed |

可选参考：单 seed BASE@32 epoch（不进主门）。

## 评估
- 主读：**outer** coverage / minADE / motion gate
- 迁移：FIT vs outer；另报 `inner`、`original_CAL`（若 harness 有）
- native 覆盖基准：outer native_coverage ≈ 0.6039（与前序一致）

## 预注册门
- **(A) seed 级**：KIN coverage ≥ native **且** motion PASS，在 ≥2/3 seed
- **(B) 均值**：3-seed mean coverage ≥ native
- **(C) 迁移**：报告 FIT−outer 覆盖差；若 |FIT−outer|≤3pp 且 outer 达标，记 migration_ok

潜力结论：
- 若 (A) 或 (B) 命中 → **potential_gate HIT**（本配方有冲过 native 的上行空间）
- 若两变体均平台显著低于 native（mean cov < native−2pp）且 motion 仍 PASS → **ceiling under this recipe**（诚实写天花板）

## 不做
- 不碰 MODE_PRESERVING / P1
- 不扫 λ / 宽网格
- 不宣称模式保持已解决
