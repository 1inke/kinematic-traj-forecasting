# 资源盘点（只读引用，不改旧系列）

日期：2026-10-18（Asia/Tokyo）

## 本实验目录
`KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018/`

## 可复用数据缓存（只读）
| 路径 | 用途 |
|---|---|
| `REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz` | native K20、change_group |
| `F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz` | GT future |
| `GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz` | FIT/outer/inner |
| `GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz` | past(8,2) 朝向归一化（末点原点）|

## 指标定义
- good：ADE≤0.2 且 FDE≤0.4（与 F16/U0/P1 质量阈值一致）
- 运动学 PASS：五指标相对同批 native 倍率 ≤ `motion_factor=1.01`

## 环境
- `<PYTHON_ENV_ROOT>\envs\astra2\python.exe` + CUDA RTX 5070 Laptop
