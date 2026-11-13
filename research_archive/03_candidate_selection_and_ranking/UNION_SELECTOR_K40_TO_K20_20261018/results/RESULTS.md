# UNION_SELECTOR_K40_TO_K20_20261018 结果

日期：2026-10-18

## 一句话
**gate_HIT = True**：GT-free 主方法 `cond_boost_11_9` 在 outer 上相对原生 **+4.48pp**（cov 0.6486 vs native 0.6039），运动学 3/3 PASS；保留了 union(+9.4pp) 增益的约一半。

## 门控
- gate_HIT (+4pp & motion PASS): **True**
- gate_STRETCH (+7pp 或距 union≤2pp): **False**（距 union 仍约 5.0pp）
- oracle K20 上界 = union K40 = **0.6982**（说明池子信息够，关键是排序）

## Seed 表
| seed | native | kin | union40 | oracle K20 | best GT-free | best_cov | lift_pp | motion | HIT |
|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| 42 | 0.6039 | 0.5652 | 0.6966 | 0.6966 | cond_top_11_9 | 0.6444 | 4.06 | PASS | True |
| 43 | 0.6039 | 0.5652 | 0.6986 | 0.6986 | cond_boost_11_9 | 0.6522 | 4.83 | PASS | True |
| 44 | 0.6039 | 0.5633 | 0.6995 | 0.6995 | cond_boost_11_9 | 0.6522 | 4.83 | PASS | True |

## 主方法明细：`cond_boost_11_9`（三 seed 同一方法）
- per-seed cov: [0.6415458937198067, 0.6521739130434783, 0.6521739130434783]
- mean lift: **+4.48pp**
- motion_pass: 3/3

## 方法要点（GT-free）
1. 复用 SPEED_CLAMP 的 KIN ckpt，拼 native∪KIN 共 40 候选
2. FIT 上用 past 条件特征 + HistGradientBoosting 学 soft quality 分数（不用 outer GT）
3. 选 11 native + 9 KIN：native 按分数取 top；KIN 分数 × 对已选 native 端点距离（互补增强）
4. 纯几何 FPS / 盲目 swap 会严重掉覆盖；必须靠可迁移排序

## Top GT-free（mean）
| method | cov_mean | lift_pp | motion_pass |
|---|---:|---:|---:|
| cond_boost_11_9 | 0.6486 | 4.48 | 3 |
| cond_top_11_9 | 0.6473 | 4.35 | 3 |
| cond_swap_calib | 0.6464 | 4.25 | 3 |
| cond_swap_8 | 0.6461 | 4.22 | 3 |
| cond_top_10_10 | 0.6457 | 4.19 | 3 |
| cond_boost_12_8 | 0.6454 | 4.15 | 3 |
| cond_boost_10_10 | 0.6451 | 4.12 | 3 |
| cond_quota_10_5 | 0.6448 | 4.09 | 3 |
| cond_global_top20 | 0.6444 | 4.06 | 3 |
| cond_top_12_8 | 0.6444 | 4.06 | 3 |
| cond_quota_6_6 | 0.6441 | 4.03 | 3 |
| cond_quota_8_4 | 0.6438 | 3.99 | 3 |

## 与前序关系
- HYBRID union K40：+9.4pp（本实验上界）
- KIN+clamp 单模 K20：~0.565，低于原生
- 本实验：K20 选子集，**+4.5pp vs native**，过预注册弱门；未到 STRETCH

## 下一步建议
- 主线：把 HGB/`cond_boost_11_9` 固化为部署选择器，或再攻 STRETCH（更好 listwise 排序 / 网络 embedding）
- 辅线：failure-only residual（NATIVE_ANCHORED）可并行，但不挡本结论
- 不要回到 MODE_DIVERSE 当主线
