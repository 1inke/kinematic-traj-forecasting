# 锁源前夹具修复

第一次预检已通过起点/source/ID/8真实GPU U参数及optimizer逐位复现，随后FD夹具的前8个FIT窗口缺少rescued保护组，尚未执行FD或G0。失败源码与trace在failures/preflight1_FD_group_support完整保留。

只把双组R导数夹具改为FIT固定membership中每组按原索引前4窗，确定后不按FD结果选择；不是G0样本，不影响support-step-v1全4096采样、24FD数值测试/eps/容差、任何工程或科学门。全部预检重新执行后才锁源。GPU Tensor scalar reporting warning为诊断显示，不是训练失败。
