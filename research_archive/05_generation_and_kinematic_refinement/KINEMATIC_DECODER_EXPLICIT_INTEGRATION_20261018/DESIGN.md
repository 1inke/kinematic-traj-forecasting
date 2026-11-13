# 设计：显式运动学解码 vs 直接位置解码

日期：2026-10-18｜最小可证伪首跑（smoke + short）

## 1. 科学问题

相同观测、K=20、训练预算下：

> 解码强制「预测矢量运动量 → 运动学积分」是否比直接输出位置提高运动学合格率，且 coverage / minADE 不明显变差？

**不声称**单独解决模式保持或 P1 责任×保护。

## 2. 两臂
| 臂 | 解码 |
|---|---|
| BASE `direct_position` | 直接输出 `p_{1:H}` |
| KIN `kinematic_integration` | 输出 `(v̂,â,Δp̂)`，轨迹仅由 `â` 积分 |

## 3. 离散方程（KIN）
`dt=0.4`，`H=12`，`p_0=0`（朝向归一化末观测）

```
v_0 = (p_0 - p_{-1}) / dt
v_t = v_{t-1} + dt * â_t
Δp_t^{phys} = dt * v_t
p_t = p_{t-1} + Δp_t^{phys}
```

一致性辅助：`L_cons = mean(|v̂-v_int| + |Δp̂-Δp_phys|)`（不替代积分）。

## 4. 门（short，seed=42）
弱阳性需同时：① KIN 运动学优于 BASE（PASS vs FAIL，或同 PASS 且 accel_mean 相对 native 倍率低 ≥5%）；② KIN coverage ≥ BASE−1pp；③ KIN minADE ≤ BASE×1.10。
阴性则停，不扫参。
