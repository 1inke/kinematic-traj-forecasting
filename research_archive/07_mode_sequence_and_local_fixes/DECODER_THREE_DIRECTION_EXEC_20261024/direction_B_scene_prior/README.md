# 方向B：历史终点先验零训练探针

状态：`CODE_PREPARED_NOT_LAUNCHED`。`EndpointPrior`显式继承既有`HistogramPrior`，覆盖旧256桶归一化与fallback，实现257完备桶、原8臂、每层target-track purge、严格同序列已完结时间资格、非同序列冷启动、稳定top4与SHA256置换种子。预测query API没有GT参数。

`prepare_records.py`要求显式raw-world NPY及SHA；只读允许FIT/inner ID的8帧世界观测，用past-only刚体对齐校验缓存坐标，从既有已锁GT取各历史片段自身局部终点。没有世界坐标/不明索引/无法验证变换时阻塞，不能伪造cell或时间。

```powershell
python -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_B_scene_prior/prepare_records.py --raw-world-npy <已登记原世界观测NPY> --raw-sha256 <真实SHA> --output <B目录下的新records.json> --estimated-peak-gib <inventory E>
python -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_B_scene_prior/probe.py --records <records.json> --output <B目录下的新execution> --estimated-peak-gib <inventory E>
```

当前未运行以上命令。B无模型、solver、训练或teacher。完整原始NLL、257 counts、支持数/层级/冷启动/有效性、endpoint冗余、8臂×查询×FIT源ID bitset均保存，避免把巨量Python源ID对象无限常驻。源码/records源核验和算术报告不冒称独立完整eligibility第二实现，正式执行前应审查记录来源。

静态主人口497（native miss331）；未知因果时间不丢ID，causal子集描述且不能替完整497放行。任一静态/因果过门只允许另设计路径生成，绝不自动训练生成器。
