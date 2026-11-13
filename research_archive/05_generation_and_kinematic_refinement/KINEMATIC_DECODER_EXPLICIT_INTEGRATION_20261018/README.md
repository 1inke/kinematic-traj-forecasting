# 显式运动学解码实验（2026-10-18）

用户授权的新独立实验：**解码阶段显式预测 (速度, 加速度, 位移) 矢量，再运动学积分成轨迹**，对照直接位置输出。

## 目录
- `DESIGN.md` — 可证伪问题、两臂、方程、门控
- `INVENTORY.md` — 只读依赖路径
- `kdec/` — 实现（积分 / 模型 / 数据 / 训练）
- `tests/` — 积分器单测
- `results/` — unit / smoke / short 产物
- `NEXT.md` — 全量 3-seed 等后续

## 快速跑
```powershell
cd "<WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018"
<PYTHON_ENV_ROOT>\envs\astra2\python.exe -B kdec\run_all.py
```

## 不做什么
- 不重开 P1/P2，不改 `MODE_PRESERVING_*` 结果
- 不扫 λ / 隐层；阴性即停
- 不声称单独解决模式保持或覆盖迁移
