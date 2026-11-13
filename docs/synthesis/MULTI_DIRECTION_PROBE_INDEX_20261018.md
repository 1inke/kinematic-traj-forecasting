# 多方向探测索引 MULTI_DIRECTION_PROBE 20261018

用户授权：在 kinematic-traj-forecasting 下新开方向试潜力；不改 MODE_PRESERVING / P1。

## 对照基准
| 基准 | outer cov | motion |
|---|---:|---|
| native K20 | 0.604 | — |
| KIN+clamp 8ep（前序强结果） | 0.565 | PASS 3/3 |
| KIN potential 32ep 天花板 | ~0.565 | PASS |

## 本轮方向

| 方向 | 文件夹 | 假设 | 关键结果 | 门 |
|---|---|---|---|---|
| **A** 原生锚定残差 | `NATIVE_ANCHORED_KIN_RESIDUAL_20261018` | native+有界运动学残差 | cov **0.605**（+0.13pp vs native），motion 3/3；2/3 beat_native | **HIT（弱）** |
| **B** 并集篮子 | `HYBRID_NATIVE_KIN_UNION_20261018` | native∪KIN 互补 | union **0.698**（**+9.4pp**），kin_only≈98 | **HIT（强）** ※K=40 |
| **C** 多模式 KIN | `MODE_DIVERSE_KINEMATIC_20261018` | mode latent 避 I1 塌缩 | cov **0.547** < KIN 天花板 | **MISS** |

## 谁赢了
1. **明确超过原生（并集设定）**：B  
2. **单模型 K20 刚贴/略超原生**：A（薄增益，不稳）  
3. **未超 KIN 天花板**：C  

## 建议下一主线
**B 的选择器/槽位融合**（40→20，保住并集增益）为主；A 的 failure-only 残差为辅。C 停。

## 一句话（中文）
并集证明原生与运动学再生高度互补（+9pp）；残差锚定能刚贴原生但很薄；多模式解码这版没超过 KIN+clamp。


## 后续主线（已执行）
| 方向 | 文件夹 | 结果 | 门 |
|---|---|---|---|
| **B→选子** K40→K20 | `UNION_SELECTOR_K40_TO_K20_20261018` | GT-free `cond_boost_11_9` cov **0.649**（**+4.48pp** vs native），motion 3/3；oracle=union 0.698 | **HIT**（STRETCH 未过） |
