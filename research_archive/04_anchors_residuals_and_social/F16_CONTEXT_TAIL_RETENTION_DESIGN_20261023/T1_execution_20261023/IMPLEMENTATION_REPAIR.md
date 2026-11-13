# 预检v1→v2保存路径修复

第一次实际预检在P_relative smoke的完整质量/聚簇/五motion和独立几何审核之后，进入正式run的旧A固定人口补充时失败：smoke目录没有seed，不能解析为生产run。失败trace及当时代码完整保留在failures/preflight1_smoke_arm_path。

唯一有效计算分支修复：`run.name.startswith('P_relative')`改为`run.name.startswith('P_relative_seed')`，只有正式run进入旧A冻结subset补充。正式run本来名称就带seed，其全部科学运算不变；预检smoke仍完整构造/保存/读回和独立质量、GTfree原聚簇、pooled五motion、几何分解，只不冒充完整18909窗旧A分组比较。

source锁前完成修复，继承旧548digest不变；源清单扩入真实失败代码和trace。全实际预检重新运行，不沿用半套PASS，不涉及模型/公式/优化/阈值变化，没有正式worker或训练启动。
