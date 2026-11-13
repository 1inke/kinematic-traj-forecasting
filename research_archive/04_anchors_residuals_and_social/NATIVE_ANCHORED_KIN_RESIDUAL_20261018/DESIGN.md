# 设计：原生锚定运动学残差

日期：2026-10-18

## 假设
不替换原生 K20 篮子；在原生候选上预测**有界运动学残差**（加速度→积分位移），软限幅后叠加：
`paths = native + integrate(tanh(a_res) * a_res_max)`。

相对纯 KIN 再生：保留原生支撑；相对 P1 保护臂：允许可学习修正但幅度有界。

## 臂
- **ID**：原生原样（对照）
- **RES**：原生 + 残差（主臂）
- **KIN**（可选参考）：纯运动学再生

## 预注册门
1. **beat_native**：outer cov ≥ native 且 motion PASS
2. **beat_kin_ceiling**：outer cov ≥ 0.5646（SPEED_CLAMP 均值）且 motion PASS
3. **weak_positive**：motion PASS 且 cov ≥ native−1pp 且 rescue ≥ lost

## 协议
- smoke：2ep / subset 2048
- short：8ep × seeds 42/43/44，全量 FIT
- 不改 MODE_PRESERVING / P1
