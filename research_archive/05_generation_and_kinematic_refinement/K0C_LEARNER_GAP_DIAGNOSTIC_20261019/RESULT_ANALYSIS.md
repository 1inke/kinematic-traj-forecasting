# K0-C Learner Gap Diagnostic — RESULT_ANALYSIS

- time: 2026-10-19T03:26:44+09:00 (Asia/Tokyo)
- dir: `<WORKSPACE_ROOT>\K0C_LEARNER_GAP_DIAGNOSTIC_20261019`
- scope: outer 三 seed；只读回放 K0-C 已接受替换 vs K1-S pred_gate / GTF 几何排序
- **BREAKTHROUGH=false**（诊断，不宣称部署门）

## 1. 集合对齐（outer）

| seed | |S∩K0C| mean | S==K0C | |GTF∩K0C| | GTF==K0C | S harm ADE frac |
|---:|---:|---:|---:|---:|---:|
| 42 | 17.09 | 0.183 | 2.01 | 0.000 | 0.024 |
| 43 | 17.36 | 0.181 | 2.46 | 0.000 | 0.054 |
| 44 | 17.15 | 0.176 | 2.49 | 0.000 | 0.033 |

## 2. K0-C 有利动作回放：S pred_gate 误拒

- 三 seed 合计 oracle 接受动作数: **1228**
- 被 S pred_gate 误拒: **1007** → 误拒率 **0.8200**
- 被 S pred_gate 接受: **221** → 接受率 **0.1800**
- 拒绝子句计数（可重叠）: L>0.01=636, dA<0=816, dF<0=909, 无(dC>0|dJ>0)=551

| seed | n_actions | misreject | accept |
|---:|---:|---:|---:|
| 42 | 388 | 0.9175 | 0.0825 |
| 43 | 448 | 0.6674 | 0.3326 |
| 44 | 392 | 0.8980 | 0.1020 |

## 3. 同一动作：GTF 几何分是否选到

- motion-PASS 可排名的 oracle 动作: **1228**
- GTF top1 命中: **8** → hit **0.0065** / misrank(非top1) **0.9935**

| seed | ranked | top1 | top3 | mean rank | misrank |
|---:|---:|---:|---:|---:|---:|
| 42 | 388 | 2 | 3 | 144.42 | 0.9948 |
| 43 | 448 | 2 | 4 | 167.38 | 0.9955 |
| 44 | 392 | 4 | 6 | 150.20 | 0.9898 |

## 4. S 实际接受但相对 native 损害 ADE

- seed42: harm_ADE=0.024, help=0.085, swaps@harm=4.6, swaps@help=5.920454545454546
- seed43: harm_ADE=0.054, help=0.168, swaps@harm=4.339285714285714, swaps@help=5.597701149425287
- seed44: harm_ADE=0.033, help=0.113, swaps@harm=6.647058823529412, swaps@help=5.230769230769231

## 5. 训练状态 S0/S4/S12 vs 部署轨迹

Train (seed42 为代表，三 seed 结构相同):
- S0: n=131072, L_pos=0.0154, dC_pos=0.0057, dA_pos=0.0262
- S4: n=131072, L_pos=0.0686, dC_pos=0.0081, dA_pos=0.0348
- S12: n=131072, L_pos=0.1350, dC_pos=0.0138, dA_pos=0.0444
Deploy S n_swaps 分布 (三 seed):
- seed42: mean=2.64, eq0=0.303, 1-3=0.459, 4-7=0.137, ≥8=0.100
- seed43: mean=2.45, eq0=0.282, 1-3=0.500, 4-7=0.154, ≥8=0.065
- seed44: mean=2.62, eq0=0.290, 1-3=0.504, 4-7=0.133, ≥8=0.072

Train samples actions equally at S0/S4/S12; deploy S mean swaps~2.5 with ~30% zero. Train S_L_pos rises sharply S0->S12, so high-swap states over-represented in lost-label training.

## 6. 判断

1. K0-C 有利替换绝大多数被 **S pred_gate 误拒**（见上误拒率）——与 Branch C「过保守」一致，且把「学不到 oracle 动作」定位到门控而非动作空间为空。
2. GTF 几何排序对 oracle 动作的 top1 命中很低（见 misrank）——解释为何纯几何 one-factor 无法复现 K0-C。
3. 训练在 S4/S12 上 L 正例远密于 S0，而部署停在低 swap；状态偏移可能放大丢失恐惧（L）预测。
4. **不**据此重训 HGB / 开 K2 / 扫配额；BREAKTHROUGH=false。

## 产物
- `DIAGNOSTIC.json`
- `seed{42,43,44}_diag.json`
- `seed{42,43,44}_oracle_action_preds.npz`


## 7. original_CAL KIN cache（工程收尾）

- 状态: **OK**（三 seed 冻结 no_grad CPU forward）
- shape: (2879, 20, 12, 2)
- 写出: UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_original_CAL_seed{42,43,44}.npy 与 CPV rtifacts/v1/ 同名副本
- 详见 CAL_CACHE_STATUS.json
- **未**做选择器训练 / 未宣称迁移门通过

## Standby

诊断完成 + CAL cache 已补。BREAKTHROUGH=false。standby=yes（等用户或硬门突破再动）。
