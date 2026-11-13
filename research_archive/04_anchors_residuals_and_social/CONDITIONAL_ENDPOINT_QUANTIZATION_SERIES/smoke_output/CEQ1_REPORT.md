# CEQ1 条件端点分布量化实验

- 范围：ETH seed 1，固定 validation IDs，不读取测试集。
- 候选族：从训练折相似历史对应的真实未来构成条件经验分布，再按概率质量量化为 20 条候选。
- 分配原则：weighted Lloyd 最小化条件分布的量化误差；没有候选间排斥项、固定方向模板或硬配额。
- 评分：每个候选的 cluster probability mass；最高分与候选生成来自同一分布。
- 校准锁定参数：`{"neighbours": 64, "temperature": 1.0, "components": 20, "lloyd_iterations": 8}`。
- 判定：**SMOKE_ONLY_NO_SCIENTIFIC_VERDICT** — This 64-sample execution checks the pipeline only; run the fixed full validation protocol for a scientific decision.

## Validation 指标

| Family | Subset | N | J | minADE | minFDE | Selected ADE | Regret | Spread | Support@1m | R@1 | R@3 | Duplicate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c00 | all | 64 | 0.1478 | 0.0831 | 0.1294 | 0.1712 | 0.0881 | 1.7671 | 8.234 | 0.438 | 0.656 | 0.0114 |
| c00 | high_motion | 20 | 0.2302 | 0.1362 | 0.1880 | 0.2262 | 0.0900 | 1.9607 | 7.200 | 0.000 | 0.350 | 0.0089 |
| c00 | non_high_motion | 44 | 0.1103 | 0.0589 | 0.1027 | 0.1461 | 0.0872 | 1.6791 | 8.705 | 0.636 | 0.795 | 0.0126 |
| ceq1 | all | 64 | 0.2170 | 0.1244 | 0.1852 | 0.3497 | 0.2253 | 1.2409 | 12.047 | 0.156 | 0.266 | 0.1621 |
| ceq1 | high_motion | 20 | 0.2653 | 0.1561 | 0.2185 | 0.4545 | 0.2984 | 1.5654 | 10.350 | 0.150 | 0.250 | 0.0855 |
| ceq1 | non_high_motion | 44 | 0.1951 | 0.1100 | 0.1701 | 0.3021 | 0.1920 | 1.0935 | 12.818 | 0.159 | 0.273 | 0.1969 |

## CEQ1 minus C00 paired bootstrap

| Subset | Metric | Delta | Relative | 95% CI |
|---|---|---:|---:|---:|
| all | j | 0.06928 | 46.89% | [0.01610, 0.12751] |
| all | minade | 0.04135 | 49.78% | [0.02376, 0.06876] |
| all | minfde | 0.05586 | 43.18% | [0.01363, 0.09492] |
| all | selected_ade | 0.17854 | 104.31% | [0.10274, 0.27211] |
| all | regret | 0.13719 | 155.75% | [0.05867, 0.23556] |
| all | endpoint_spread | -0.52615 | -29.78% | [-0.65484, -0.41277] |
| all | support_1p0 | 3.81250 | 46.30% | [2.80039, 4.75312] |
| all | oracle_recall_1 | -0.28125 | -64.29% | [-0.40273, -0.14766] |
| all | oracle_recall_3 | -0.39062 | -59.52% | [-0.53125, -0.29688] |
| all | duplicate_pair_rate | 0.15066 | 1317.99% | [0.10532, 0.20481] |
| high_motion | j | 0.03518 | 15.28% | [-0.00333, 0.09577] |
| high_motion | minade | 0.01989 | 14.61% | [-0.00652, 0.04941] |
| high_motion | minfde | 0.03056 | 16.26% | [-0.01486, 0.08699] |
| high_motion | selected_ade | 0.22834 | 100.95% | [0.03505, 0.48530] |
| high_motion | regret | 0.20844 | 231.60% | [0.00207, 0.33744] |
| high_motion | endpoint_spread | -0.39537 | -20.16% | [-0.57466, -0.19289] |
| high_motion | support_1p0 | 3.15000 | 43.75% | [1.24500, 4.60500] |
| high_motion | oracle_recall_1 | 0.15000 | NA% | [0.05000, 0.28875] |
| high_motion | oracle_recall_3 | -0.10000 | -28.57% | [-0.35000, 0.13875] |
| high_motion | duplicate_pair_rate | 0.07658 | 855.88% | [0.05132, 0.11750] |
| non_high_motion | j | 0.08479 | 76.87% | [0.03784, 0.16189] |
| non_high_motion | minade | 0.05111 | 86.71% | [0.02233, 0.07895] |
| non_high_motion | minfde | 0.06736 | 65.56% | [-0.00907, 0.16494] |
| non_high_motion | selected_ade | 0.15591 | 106.68% | [0.07249, 0.23476] |
| non_high_motion | regret | 0.10480 | 120.18% | [0.03848, 0.18845] |
| non_high_motion | endpoint_spread | -0.58560 | -34.88% | [-0.71214, -0.44992] |
| non_high_motion | support_1p0 | 4.11364 | 47.26% | [3.07102, 5.02500] |
| non_high_motion | oracle_recall_1 | -0.47727 | -75.00% | [-0.65909, -0.23750] |
| non_high_motion | oracle_recall_3 | -0.52273 | -65.71% | [-0.65398, -0.37386] |
| non_high_motion | duplicate_pair_rate | 0.18433 | 1467.62% | [0.12321, 0.25109] |

## Train-only calibration

| K neighbours | Temperature | Objective | minADE | minFDE | Selected ADE |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.0 | 0.438654 | 0.1566 | 0.2614 | 0.6055 |

## 判定门槛

- `generation_quality`: False
- `selected_ade_improves_2pct`: False
- `support_1m_increases_0p25`: True
- `high_motion_j_not_worse_1pct`: False
- `duplicate_pair_rate_below_10pct`: False

## Provenance

```json
{
  "experiment": "CEQ1_CONDITIONAL_EMPIRICAL_FUTURE_QUANTIZATION",
  "scope": "ETH seed 1 validation-only feasibility pilot",
  "new_neural_training": false,
  "test_paths_read_or_passed": false,
  "candidate_family": "training-only conditional empirical future distribution; weighted full-trajectory Lloyd components; component mass is score",
  "not_used": [
    "pairwise repulsion",
    "fixed global endpoint prototypes",
    "hard candidate-family quota",
    "validation future in retrieval features"
  ],
  "split": {
    "train_count": 28791,
    "validation_count": 1516,
    "validation_id_sha256": "dfd29348a6eb3d3ac601c3fae60250d86762a85fb94ea326d050d582621e870d",
    "split_file_sha256": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e"
  },
  "baseline": {
    "checkpoint": "<WORKSPACE_ROOT>\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\v2\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
    "checkpoint_sha256": "691e9f9ff6b7dc1df1c7d5f4901d97ea9885fac891099c84044bf56dd98d2341",
    "contract": {
      "decoder_class": "JointTrainableV2Decoder",
      "decoder_fingerprint": "4bec38da2b5907d9",
      "num_modes": 20,
      "decoder_experiment": "v2_joint_wide_endpoint_correction",
      "encoder_experiment": "none",
      "classification_weight": 0.5,
      "loss_class": "ModeQueryLoss",
      "fusion_alpha": 0.3,
      "oracle_coverage_weight": 0.2,
      "oracle_coverage_temperature": 0.1,
      "oracle_coverage_fde_weight": 0.25,
      "oracle_coverage_warmup_epochs": 0.0,
      "oracle_coverage_normalized": false,
      "scene_sampling": "none",
      "motion_difficulty_sampling": false,
      "motion_difficulty_strength": 0.8,
      "ema_decay": 0.995,
      "augmentation": {
        "rotation": false,
        "reflection": true,
        "speed_min": 0.9,
        "speed_max": 1.1,
        "probability": 0.5,
        "curvature_max": 0.0
      },
      "future_difficulty_weight": 0.0,
      "focal_heading_normalize": true,
      "velocity_loss_weight": 0.0,
      "acceleration_loss_weight": 0.0,
      "masked_history_weight": 0.0,
      "masked_history_probability": 0.15,
      "equivariance_weight": 0.0,
      "training_only_auxiliary_parameters": 0,
      "checkpoint_criterion": "oracle_combo",
      "hierarchical_decoder": {
        "cv_reference": false,
        "anchor_steps": [
          4,
          8,
          12
        ],
        "top_m": 20,
        "use_refiner": true
      },
      "parameter_counts": {
        "total": 321988,
        "trainable": 321411,
        "decoder": 54829
      }
    }
  },
  "smoke": true,
  "device": "cuda:0",
  "feature_dimension": 42,
  "calibration": {
    "fit_count": 2048,
    "calibration_count": 64,
    "seed": 20260922,
    "objective": "mean(J) + 0.25 * mean(selected_ADE)",
    "selected": {
      "neighbours": 64,
      "temperature": 1.0,
      "components": 20,
      "lloyd_iterations": 8
    }
  },
  "validation_count_evaluated": 64,
  "motion_threshold_train_q75": 0.1424953192472458,
  "verdict": {
    "code": "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT",
    "explanation": "This 64-sample execution checks the pipeline only; run the fixed full validation protocol for a scientific decision.",
    "gates": {
      "generation_quality": false,
      "selected_ade_improves_2pct": false,
      "support_1m_increases_0p25": true,
      "high_motion_j_not_worse_1pct": false,
      "duplicate_pair_rate_below_10pct": false
    }
  }
}
```
