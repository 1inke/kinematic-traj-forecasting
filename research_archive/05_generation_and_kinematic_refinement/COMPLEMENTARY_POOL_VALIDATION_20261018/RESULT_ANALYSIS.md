# 互补候选池验证：完整执行分析报告

日期：2026-10-18 17:13（Asia/Tokyo）。
目录：`COMPLEMENTARY_POOL_VALIDATION_20261018`。
对照合同：`GROK_KINEMATIC_REVIEW_20261018/NEXT_VALIDATION_DESIGN.md`。
**不改写**旧 UNION_SELECTOR / KIN / P1 数字。

## 执行清单状态

| 阶段 | 状态 |
|---|---|
| V0 证据冻结 | 完成 → `ORIGINAL_SELECTOR_REPLAY_UNAVAILABLE` |
| V1 锁定重现 cond_boost_11_9 | **完成**（产物齐全） |
| improve（kinsafe/natunique） | **完成**；FIT_cal 选 `kinsafe_11_9` |
| V2 条件特征 vs geometry | **完成** |
| 自改进配额 11–14 | **完成**；FIT_cal 选 `boost_14_6`（探索） |
| 自改进自适应配额 | **完成**；FIT_cal 仍偏好 `kinsafe_11_9` |

## V0
- replay：`ORIGINAL_SELECTOR_REPLAY_UNAVAILABLE`
- 无 generator 训练：True
- 缺原 HGB / selected_paths → 转入 V1 冻结重建。

## V1 锁定主方法（cond_boost_11_9）

固定：HGB 超参、good 二值标签、11 native + 9 KIN、boost=0.35、稳定并列。
FIT 内按 track 划校准集；**未**用 outer 重选主方法。

### outer 主表（三 seed 均值）

| 组合 | 覆盖 | minADE | minFDE | motion_pass |
|---|---:|---:|---:|---:|
| native20 | 0.6039 | 0.17548 | 0.29663 | 3/3 |
| kin20 | 0.5646 | 0.20866 | 0.36597 | 3/3 |
| union40 | 0.6982 | 0.15544 | 0.24098 | 3/3 |
| cond_top_11_9 | 0.6461 | 0.19776 | 0.35970 | 3/3 |
| cond_boost_11_9 | 0.6480 | 0.19494 | 0.35234 | 3/3 |

- 覆盖相对 native：**+4.41 pp**（0.6039 → 0.6480）
- minADE 相对 native：**+11.09%**（恶化=True）
- minFDE 相对 native：**+18.78%**（恶化=True）
- 判定标签：**覆盖—误差折中**
- 全部 seed×group 五运动学门：True（fails=[]）
- major_deploy_gate：False（探索性，非独立验证）

### 变化组 change（outer）
- native 覆盖均值：0.4348
- 选择器覆盖均值：0.4545
- rescue/lost per seed：[13, 17, 9] / [7, 8, 9]
- motion：['PASS', 'PASS', 'PASS']

### 探索性 CI（family8）
- **overall_coverage**: mean=0.04443648264187552 ci_family8=[0.02125006886573067, 0.0707432348098576] valid=100000
- **overall_minADE**: mean=0.01953845820800584 ci_family8=[0.00874389147175525, 0.032787427316605174] valid=100000
- **overall_minFDE**: mean=0.05592670847872713 ci_family8=[0.025303394995615746, 0.09292463722569098] valid=100000
- **overall_minJ**: mean=0.11895164704172154 ci_family8=[0.05531236875023853, 0.1967189730463166] valid=100000
- **change_coverage**: mean=0.019205853830939985 ci_family8=[-0.03463203463203463, 0.07324628496503464] valid=100000
- **change_minADE**: mean=0.06983308884526439 ci_family8=[0.03226891551232584, 0.11602008198724773] valid=100000
- **change_minFDE**: mean=0.19795228471495768 ci_family8=[0.09247171262526926, 0.32772722105228347] valid=100000
- **change_minJ**: mean=0.4243594205218112 ci_family8=[0.19983250414257098, 0.69855281771346] valid=100000

## improve（FIT_cal 选臂）
- 选中：`kinsafe_11_9`
- 规则：maximize cov*100 - 0.05*lost - 50*(ade-0.17) + change_motion_pass

### FIT_cal 排名
- kinsafe_11_9: cov=0.6140 ade=0.2102 lost=46.3 ch_pass=1 score=58.08
- natunique_11_9: cov=0.6144 ade=0.2092 lost=45.3 ch_pass=0 score=57.21
- natunique_kinsafe_11_9: cov=0.6143 ade=0.2092 lost=45.3 ch_pass=0 score=57.20
- locked_boost_11_9: cov=0.6137 ade=0.2101 lost=47.0 ch_pass=0 score=57.01

### outer 确认
- kinsafe_11_9: cov=0.6496 ade=0.1946 fde=0.3517 lost=16.7 rescue=64.0 change_cov=0.4545 ch_motion=3/3 ov_motion=3/3
- locked_boost_11_9: cov=0.6480 ade=0.1949 fde=0.3523 lost=17.0 rescue=62.7 change_cov=0.4545 ch_motion=3/3 ov_motion=3/3
- natunique_11_9: cov=0.6470 ade=0.1952 fde=0.3541 lost=18.0 rescue=62.7 change_cov=0.4493 ch_motion=2/3 ov_motion=3/3
- natunique_kinsafe_11_9: cov=0.6486 ade=0.1949 fde=0.3533 lost=17.7 rescue=64.0 change_cov=0.4493 ch_motion=3/3 ov_motion=3/3

## V2 条件特征消融

### FIT_cal
- cond_features: cov=0.6137 ade=0.2101 lost=47.0 change_cov=0.3564
- geometry_only: cov=0.6110 ade=0.2130 lost=51.0 change_cov=0.3578

### outer
- cond_features: cov=0.6480 ade=0.1949 lost=17.0 change_cov=0.4545
- geometry_only: cov=0.6428 ade=0.1950 lost=19.0 change_cov=0.4466

- outer 条件特征相对 geometry_only 覆盖差：**+0.52 pp**（条件特征略优；容量不同，解释有局限）

## 自改进：固定配额（更多原生槽）
- FIT_cal 选中：`boost_14_6`（探索；**不**替代锁定主方法）

### outer 对照（关键臂）
| 方法 | 覆盖 | minADE | minFDE | lost | rescue | change_cov | ch_motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| boost_11_9 | 0.6480 | 0.1949 | 0.3523 | 17.0 | 62.7 | 0.4545 | 3/3 |
| boost_12_8 | 0.6454 | 0.1914 | 0.3433 | 12.7 | 55.7 | 0.4545 | 3/3 |
| boost_13_7 | 0.6432 | 0.1883 | 0.3362 | 9.3 | 50.0 | 0.4532 | 3/3 |
| boost_14_6 | 0.6393 | 0.1842 | 0.3251 | 6.3 | 43.0 | 0.4506 | 3/3 |
| kinsafe_11_9 | 0.6496 | 0.1946 | 0.3517 | 16.7 | 64.0 | 0.4545 | 3/3 |
| kinsafe_12_8 | 0.6464 | 0.1911 | 0.3427 | 12.7 | 56.7 | 0.4545 | 3/3 |
| kinsafe_13_7 | 0.6444 | 0.1884 | 0.3361 | 9.3 | 51.3 | 0.4532 | 3/3 |

读法：把原生槽从 11→14，**lost 与 ADE/FDE 明显改善**，但覆盖从 ~0.648 降到 ~0.639（仍高于 native 0.604）。
这是覆盖—误差—丢失折中面，不是单点最优。

## 自改进：自适应原生配额
- FIT_cal 选中：`kinsafe_11_9`

| 方法 | outer 覆盖 | minADE | lost | change_cov |
|---|---:|---:|---:|---:|
| adaptive | 0.6454 | 0.1917 | 12.0 | 0.4545 |
| adaptive_kinsafe | 0.6464 | 0.1914 | 12.0 | 0.4545 |
| kinsafe_11_9 | 0.6496 | 0.1946 | 16.7 | 0.4545 |
| locked_11_9 | 0.6480 | 0.1949 | 17.0 | 0.4545 |

自适应配额落在 11/9 与 12/8 之间：覆盖接近锁定，lost/ADE 略好；FIT_cal 仍略偏 `kinsafe_11_9`。

## 总结论

1. **互补池成立**：union40 覆盖 0.698 vs native 0.604（+9.4pp）；K20 选择器锁定方法约 **+4.4pp**。
2. **覆盖—误差折中**：锁定 11/9 覆盖↑，但 ADE/FDE 相对 native 恶化；非全面优于 native。
3. **运动学**：本 V1 重建下 outer 全部 seed×group（含 change）**PASS**（补齐了原报告只看 overall 的缺口）。
4. **条件特征**略优于 geometry-only；值得保留，但非戏剧性。
5. **保留原生支撑**可操作：提高原生配额（14/6）或 kinsafer，能降 lost/ADE，代价是覆盖部分回吐。
6. **边界**：outer 曾参与历史选法 → 探索性阳性；未扩 generator、未大扫参；W0 非本分支前置门。
7. **建议**：主方法继续锁定 `cond_boost_11_9` 作报告基线；若产品更厌 lost/尾部误差，可在独立验证集上预注册比较 `kinsafe_11_9` 与 `boost_12_8`/`14_6`，勿用本次 outer 改主方法。

## 产物入口
- `results/v0/MANIFEST.json`
- `results/v1/SUMMARY.json` + `artifacts/v1/*_bundle.npz` / `hgb_seed*.joblib`
- `results/improve/IMPROVE_SUMMARY.json`
- `results/v2/V2_ABLATION.json`
- `results/improve_quota/QUOTA_IMPROVE_SUMMARY.json`
- `results/improve_adaptive/ADAPTIVE_SUMMARY.json`
- 本文件 `RESULT_ANALYSIS.md` / `SUMMARY.md` / `STATUS.md` / `CHECKLIST.md`
