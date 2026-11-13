# 2026-10-24有限系列执行授权与修复边界

人类最新明确指令：**“自行跑这三个方向的实验，有问题自己解决。”**

该新指令覆盖此前仅代码准备状态，也覆盖A筛查通过后A-Train、C1、C2各单项的执行授权；按顺序逐项运行，不在模型内叠加机制，不再为同范围实现问题重复请求确认。A的98%/80%/15%准入门及各训练科学门不变；B只探针，不自动训练场景生成器。

原`DECODER_THREE_DIRECTION_20261023`代码及文档逐字节保留，本执行版本位于`DECODER_THREE_DIRECTION_EXEC_20261024`。源锁前已发现并修复两处预检实现问题：旧统计含float32运行buffer与float64累计证明，必须分别比较同精度字段；保存128审核需先实际NPZ保存/重读，不能只比较内存数组。两项不改任何科学模型、数据、损失、阈值或旧文件。

资源按任务inventory的1.3动态余量/Windows保留峰；CPU阶段E3GiB（NumPy/SciPy基线.3、encoder/features/native/GT与metadata.7、投影/簇/独立审核scratch1.2、IO余量.8）；GPU阶段E6GiB（旧生产private实际3.174GiB、新graph/profile与槽通信.8、独立NPZ/CI scratch1.1、IO余量.926）。不是固定6GiB门；无30min预算。磁盘总新增保守估算4GiB，需1.3倍空闲；已有约11GiB可用，运行前再以实际单次snapshot核验。

C2整模测量将实际冻结encoder接新decoder，只走观测，不运行旧decoder/scorer。MAC按实际执行矩阵乘与卷积FMA=2FLOP计数，非MAC点算/归一化/激活另列，不冒称Linear-only等于整模。B原世界观测与元数据从继承manifest定位，完整原SHA检查；不猜cell或时间。实现错误在新版本保留失败来源后修复，实质无法解释source/ID/科学/资源失败停止该项并报告，不降标准使PASS。
