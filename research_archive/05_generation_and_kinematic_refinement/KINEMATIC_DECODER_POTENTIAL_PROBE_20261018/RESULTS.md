# 结果报告 POTENTIAL_PROBE

时间：2026-10-18T13:37:31+0900（Asia/Tokyo）

## 预注册门
- (A) seed：cov≥native 且 motion PASS ≥2/3
- (B) 均值：mean cov ≥ native
- (C) 迁移：|FIT−outer|≤3pp

## 变体 LONG

- potential_gate_HIT: **False** (A=False count=0/3; B=False)
- ceiling_under_recipe: True
- coverage mean±std: 0.5614±0.0050  (native 0.6039)
- gap vs native: -4.25±0.50 pp
- minADE mean: 0.2060
- motion PASS: 3/3
- FIT−outer cov pp: -0.1531055298412428 (migration_ok=True)

| seed | outer_cov | native | gap_pp | minADE | motion | FIT_cov | CAL_cov |
|---:|---:|---:|---:|---:|---|---:|---:|
| 42 | 0.5643 | 0.6039 | -3.96 | 0.2054 | PASS | 0.5662 | 0.5155 |
| 43 | 0.5643 | 0.6039 | -3.96 | 0.2047 | PASS | 0.5618 | 0.5085 |
| 44 | 0.5556 | 0.6039 | -4.83 | 0.2080 | PASS | 0.5514 | 0.5255 |

## 变体 LONG_RELAX

- potential_gate_HIT: **False** (A=False count=0/3; B=False)
- ceiling_under_recipe: True
- coverage mean±std: 0.5646±0.0030  (native 0.6039)
- gap vs native: -3.93±0.30 pp
- minADE mean: 0.2060
- motion PASS: 3/3
- FIT−outer cov pp: -0.4134677068890591 (migration_ok=True)

| seed | outer_cov | native | gap_pp | minADE | motion | FIT_cov | CAL_cov |
|---:|---:|---:|---:|---:|---|---:|---:|
| 42 | 0.5652 | 0.6039 | -3.86 | 0.2059 | PASS | 0.5650 | 0.5155 |
| 43 | 0.5614 | 0.6039 | -4.25 | 0.2075 | PASS | 0.5552 | 0.5252 |
| 44 | 0.5671 | 0.6039 | -3.67 | 0.2046 | PASS | 0.5611 | 0.5297 |

## STOP
- potential_gate_HIT: **False**
- ceiling_under_recipe: **True**
- best_variant: LONG_RELAX

## INFERENCE
潜力门未命中；两变体均平台低于 native≥2pp（best=LONG_RELAX mean_cov=0.5646 vs native≈0.6039）。本配方下冲 native 的上行空间有限。

## 补充事实

### mid-eval（未进主门，仅审计）
- LONG seed42：ep8 cov=**0.5836** → ep16 0.547 → ep32 0.564（峰值仍 < native 0.604）
- LONG_RELAX 各 seed 终局与 ep8–24 均在 0.54–0.57 带，无穿越 native

### BASE_REF seed42 @32ep（参考，非主臂）
- outer cov **0.5739**，minADE 0.191，motion **FAIL**（accel_mean ratio≈1.12）
- 说明：更长训直接位置也能抬覆盖，但运动学仍不过；KIN 用结构换 PASS，覆盖略低于该 BASE 参考

### 与 SPEED_CLAMP 短训对比
| 协议 | KIN outer cov 均值 | motion | vs native |
|---|---:|---|---|
| 8ep clamp | 0.5646 | PASS 3/3 | −3.9pp |
| 32ep LONG | 0.5614 | PASS 3/3 | −4.3pp |
| 32ep LONG_RELAX | 0.5646 | PASS 3/3 | −3.9pp |

→ **加长/略松限幅未抬升天花板。**
