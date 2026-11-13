# 当前交付与未来启动边界

## 本次已做（代码准备，不是训练）

- 两个独立方向、各有accuracy_only和accuracy_dpp三seed对照。
- 真实联合目标、CVAE训练posterior与部署prior/DSF严格分开。
- 配对初值、逐epoch ID顺序、optimizer与ALL RNG/cursor恢复代码，完成SHA兼容skip。
- FIT/dev原split来源、缓存、calibration与旧锁源只读SHA绑定。
- CPU合成NumPy q/EC、同slot损失值与解析路径梯度、CVAE+DSF完整参数链固定2方向×2eps FD。
- FD记录winner、ReLU、运动超限/零norm、Gaussian clamp活动面，不缩eps找有利结果。
- 默认入口plan、显式--execute依然被当前hold拒绝，无CUDA上下文、无optimizer更新。

## 尚未实际执行，不能伪PASS

- 当前版本的真实CUDA分阶段梯度/原模型zero-nativeidentity。
- 各family/arm真实513 FIT两epoch34update在19中断、全部状态/RNG EXACT。
- 各arm实际128非零保存原完整K20聚簇/pdist/五motion与独立q/EC重算。
- 当前GPU内存峰、吞吐、磁盘准入与生产ETA；没有先占用或锁定任何内存。
- 12个正式模型训练、26人口最终评价、family30科学CI。
- 全模型原始观测端到端成本与完整参数计数；CVAE缓存模块参数不是全模型。

只有后续明确启动指令才能改变control；然后执行preflight_real.py并在新executions目录建生产源锁。训练命令只跑指定family或all，不自动评价/开放held；全部12final锁好后才显式evaluate。

现有脚本未实现OS任务注册或自动监控，本次不新建任何后台任务。若后续需脱离对话运行，先用当前正式预检收据和pre-Popen资源准入建立独立launcher，再做唯一实际身份startup；不能把脚本存在冒称已经启动。

部分population已有文件而done未完整时拒绝覆盖，需新恢复namespace保留来源；不得删除失败文件、重训已完成兼容模型或更改locked数学。源、ID、数值不一致或资源真实不足应保留证据STOP，不修改科学门。
