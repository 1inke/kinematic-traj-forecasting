# 设计：KIN + 速度/加速度软限幅

## 问题
上一版 KIN 三 seed 覆盖与 ADE 均优于 BASE，但 motion 门因 speed_p95 或 accel_mean 略超 native×1.01 而未过。

## 假设
在积分递推中对 `|a|`、`|v|` 做可微软限幅（参考 FIT-native speed_p95 / accel_p95），并加轻量运动裕度损失，可使 ≥2/3 seed 达到预注册 weak_positive，而不显著损害 coverage/ADE。

## 两臂
| 臂 | 说明 |
|---|---|
| BASE | 与上一版相同：直接位置解码 |
| KIN | 显式积分 + **速度/加速度软限幅** + 运动裕度损失 |

## 限幅方程
`dt=0.4`，`H=12`，`p_0=0`，`v_0` 来自历史。

```
a_t = soft_clamp_norm(a_hat_t, a_max)
v_t = soft_clamp_norm(v_{t-1} + dt * a_t, v_max)
p_t = p_{t-1} + dt * v_t
```

其中 `soft_clamp_norm(x, m) = x * (m / max(||x||, m))`（可微近似用 `||x||_eps`）。

默认（Round1，预注册一版，不扫）：
- `v_max = FIT_native_speed_p95 * 0.995`
- `a_max = FIT_native_accel_p95 * 0.90`
- 运动裕度损失权重 `motion_w=0.05`，目标倍率 0.995

## 门（同上一版）
weak_positive：KIN motion 优于 BASE（PASS vs FAIL，或双 PASS 且 accel 优 5%）且 coverage≥BASE−1pp 且 minADE≤BASE×1.10。

**强结果目标**：≥2/3 seed weak_positive；更强则 KIN motion PASS 且 coverage≥native(0.604)。
