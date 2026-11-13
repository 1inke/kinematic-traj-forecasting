# T1执行：A_replay vs P_relative

用户2026-10-23“按照原计划执行”授权。D0/D1入口已满足。最多6fresh runs，每臂42/43/44各8epochs；固定epoch8、不扫描系数或选择checkpoint。

与父设计完全一致：82008参数/210观测输入/全K20直接轨迹；A原Qhard+.05 endpoint线性hinge，P仅额外.2 mean relu(Jstudent-detach Jnative)。原生参考训练float32、评价float64。仅FIT18909训练，inner1856事后评价，不outer/CAL。原F16/decoder/scorer不实例化或更新，不冒称重新GPU冻结回放。

旧统计、初始化、八轮顺序逐字节复制锁定。A fresh复现旧A的最终路径bitwise及历史共同字段；不符就CONTROL_REPRODUCTION_UNRESOLVED停止保留证据，不重复训练凑匹配。

只有自己的完整实际GPU/双eps活动面FD/双方513FIT+12inner两轮全部optimizer与RNG恢复/实际hash128全聚簇五motion和几何独立审核/20k bootstrap第二实现/source-resource预检通过才独立OS启动。资源E6GiB来自旧生产峰与明确增量inventory，不是固定6GiB门，使用1.3动态剩余与Windows保留峰政策。

新增全部只本目录。引用旧core经过只读检查，无legacy main/train顶层写操作；D0计算函数只读AST白名单提取，独立审核另一实现。原548源digest继承不覆盖并扩D0证据与当前实现。运行控制与源清单分离。

次级family16 reference-success lost率：P-A以A成功为参考，报告P丢失其成功的比例（A自身对应0）；P-native同理。工程保留门另独立比较P与A对native成功的丢失率，避免混淆分母。全部16项和主R配对track+seed/fixedseed敏感性保留，源/ID/科学标准不放宽。

生产完成需worker+exit双SUCCESS、六checkpoint48epoch3552steps、12科学人口输出全部SHA/完整原GTfree聚簇/五pooledmotion/同slot几何和GT时序独立审核、20k CI完整复核、源终SHA及资源。保存算术不是模型/梯度全回放；完成计算后仍需同助手反方与报告。无自动B/E/scorer或新变体。
