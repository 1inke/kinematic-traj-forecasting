# 设计：多模式运动学解码（避 I1 塌缩）

日期：2026-10-18

## 假设
4 个 mode latent × 每模式 5 slot → 运动学积分；损失 = winner-of-K + **模式中心**多样性，**不是**全候选同 GT MSE。

## 门
- beat_kin_ceiling：cov≥0.5646 且 motion PASS
- beat_native：cov≥native 且 motion PASS
