# 执行前实现修复清单（科学结果前锁定）

上位合同、数据人口、模型维度/初值/8轮shuffle、优化器、任务损失与所有阈值不变。10-23完整代码包保留，此处为新的10-24执行版。

1. 旧`statistics.npz`的`mean/std`是float32、`mean_float64/std_float64`才是累计证明。预检分别比较float32 EXACT和float64原1e-11，而非错误跨精度比较；训练仍用原float32 buffer。
2. 预检128样例先真实保存NPZ并读回给独立审核，不只核对内存；A-Screen所有12保存人口增加原独立完整聚簇/五motion复算。
3. 冻结encoder profile的ASTRA隐式`models`包与新`models.py`冲突。两次预检导入失败源码与错误在`failures/profile1_models_namespace`、`profile2_namespace_resolution`保留；显式namespace包路径隔离后实际256 FIT观测重放PASS。新decoder class引用不变，无旧权重更新。
4. B世界→缓存局部对齐使用proper-rotation Kabsch，避免直线rank-one SVD的任意反射补全造成伪失败；仍按原2e-5坐标门。实际20765登记最大误差8.0947e-7；没有未来参与对齐。
5. C1_hard保留旧S1/T1 identically-zero无teacher分支图以严格复现控制；数值损失和梯度恒等不变。C1_awta、C2和A科学因素不改。
6. 新增实际encoder+decoder整模operator级FLOP/MAC计数与真实batch1/256延迟脚本，不将旧Linear-only下界冒充整模。旧decoder/scorer在forward prehook处截断，不执行；全部旧参数/buffer/布尔量前后EXACT。
7. 最新人类指令是执行授权，已覆盖各包，不让历史CODE_ONLY状态/重复人类询问阻塞执行。仍串行单因素，A按原筛查门条件触发，不添加第四局部变体。

初次研究反证：最容易的误报是把实际接口/保存审核通过当作科学改善。本系列所有预检与正式结果分文件，最终双SUCCESS后仍需同助手人工反证总结（非外审）；任何科学阴性按原门保留，不改标准。
