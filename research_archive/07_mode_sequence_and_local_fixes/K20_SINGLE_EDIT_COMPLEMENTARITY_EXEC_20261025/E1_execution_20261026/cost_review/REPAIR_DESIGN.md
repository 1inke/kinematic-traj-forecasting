# 成本补充的显式来源与失败边界

本目录是E1源锁后的有限成本扩展，非原700科学源的一部分，不修改原source_manifest及任何旧代码/模型/缓存/结果。

benchmark_raw.py在CPU第一个固定opt ID16381发现原2e-5缓存一致性界未过：native1.2040e-4、tokens5.8442e-4，保存FAILURE。该数值界不放宽，不能称CPU回放通过。

benchmark_raw_v2.py保留逐ID原2e-5 PASS/FAIL判定，只允许继续采集失败CPU名义同权重路径的描述性成本，最终状态必须PARITY_BLOCKED；这不是科学容差修改。该版本切GPU时Module.to没有更新旧ASTRA的普通device属性，temporal_encoder仍创建CPU位置编码，混设备失败。源及trace永久保留。

benchmark_raw_v3.py唯一后续实现修复：与Module.to(device)同时设置legacy net.device、net.cfg.device/device_list。这些是临时实例的运行设备属性，既有模型文件/科学方程不改。实际CPU/GPU50warm500完成，旧全部参数与buffers初末EXACT、来源文件SHA不变、无backward/optimizer。

最终GPU固定32ID的四组件缓存一致性均PASS；CPU32全部FAIL，最大native5.40972e-4/tokens8.26299e-4，past0、KIN1.90735e-6。CPU差异未被本轮解释为某个确定根因，不更换精度/backend或搜索容差使PASS；只报告CPU数值一致性阻碍。名义路径成本可描述，不能用它证明CPU复现实验输出或部署安全。

唯一参数397100>305000独立FAIL；主科学比较本身阴性。故本有限系列停止，不因上述成本失败重训或无限局部修补。REVIEW.json与COMPLETION_EXTENSION绑定本目录代码、实际结果和所有失败SHA；原700与1prelaunch扩展保持。
