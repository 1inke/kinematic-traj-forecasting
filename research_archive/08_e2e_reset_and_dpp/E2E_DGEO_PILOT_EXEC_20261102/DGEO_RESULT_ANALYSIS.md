# DGeo 四臂执行与机制评审

四fresh seed42固定8轮完成。训练/损失/EMA/数据/clip5/64D与原B0配对不变；原筛选结果见PILOT_SCREEN，原始全组与支持交换见ALL_GROUPS/PAIRED_COMPARISON。此补充只CPU保存数据，不再训练/模型前后向/全聚簇。

|arm|overall ADE|change ADE|overall coverage|change coverage|原联合门|新增机制配对点信号|
|---|---:|---:|---:|---:|---|---|
|DG-Base|0.205628|0.262689|0.460762|0.375000|False|composite control|
|DG-GeoGate|0.198403|0.268317|0.476331|0.332292|False|False|
|DG-GeoDiff|0.197951|0.266980|0.481170|0.347917|False|True|
|DG-GeoScore|0.198343|0.268498|0.473596|0.314583|False|False|

## ARS DA2（同助手，非外审）

几何+中心化是联合干预，不能分别证明哪个独因果。DG-Base相对B0同时有容量/初值/部分评分特征解耦变化。非零增益出口没有改变初始轨迹，但改变了第一次proposal_delta梯度；不能称全部优化几何保持不变。GeoScore更改新评分残差且可能经共同回归骨干影响候选集，因此共同覆盖和各自覆盖分母同时报告。

负权和off-diagonal cosine只是算子/表示描述，接近0不等有效模式、负权多不等去噪成功。实际结果优先于方向文档里未独立证实的固定数值、信息论归因或因果陈述。

## ARS DA3与最终边界（同助手，非外审）

不能用无条件p95/p99替代原条件尾部门，或用不同条件人群变化直接断言物理风险已被修好。零positive-excess参考仍NOT_ESTIMABLE，不加epsilon/floor。physical-slot相配不等同一语义路径。任何旧支持丢失均按原逐ID记录，不只报净增。

单seed42、dev相关窗/同track/反复开发、姐妹录像独立性UNKNOWN，不是独立场景泛化或显著性。此worker在四臂结束，三seed仅记满足原门后可申请自身预检/预算，无后台自动续训。原major/baseline/historycycle11/未完goal不变。
