# 成功保留参数更新实验

授权：2026-10-25用户“按设计目录推进实验”。原三份设计快照不改。

顺序：实际起点/源/ID准入与数值预检 → G0独立一步反事实 → 只有全G0门通过才G1九个配对续训 → G2保存审核。所有科学阈值原样，旧模型、结果、失败及暂停跟进不改。

U使用原AdamW完整增量；N在自身before-state匹配反事实投影增量的范数；P应用两半空间最近投影。三臂同样计算约束但仅P改方向。optimizer矩始终按原裁剪task梯度推进。float32写回可能有有界正约束残差，不冒称严格非增。

只FIT18909训练与inner1856评价。保护组是冻结FIT中native与A0共同成功/原native失败A0成功。旧A起点是epoch8/592step的模型、AdamW、全部RNG，新增loader seed2026102500+modelseed。完整网络349167参数仍超305000。

G0没有续训checkpoint选择；G1没有inner选模型、额外臂或超参搜索。每阶段真实耗时/内存预检后才安排独立按需0trigger worker及单次half→completion。资源为inventory估值×1.3动态remaining/Windows保留峰，不是固定6GiB门；无固定分钟预算。
