# D-0：新评价管线回放与精度校准

预算：0 正式训练、0 GPU 小时，仅已完成 EXP03 的三 seed outer 保存输出。此处重算不是新 S3，不将历史 A/B 当作新协议下的同批训练对照，也不解封 original_CAL 结果。

在读取结果前固定：A−native、B−native、B−A 三个既有对照；以 outer change 的全部窗口、实际 track、配对 seed42–44，20,000 PCG64(20261020) draws，每次先 track 后 seed。点估计为窗口加权的各 seed 均值。报告普通95%百分位区间和 bootstrap SE。80% MDE 使用双侧 α=.05 正态近似 `(z.975+z.8)*SE`，不是实证功效模拟，也不是可迁移到任意新方法的统一功效保证。

独立审计从保存路径重新计算质量、Q-winner 同路径误差、完整 GT-free complete-link 聚簇与五项 pooled motion，并绑定冻结 native/GT/group 的 SHA 与 ID；对照旧报告双方共同字段，新增字段不伪造历史值。所有旧输入前后 SHA 不变。

处置：结果交作者选择 MDE 与统计约定；重叠审计 C3/C6 未关闭前不冻结协议、不做 A/B 正式训练。历史实验判定不追溯改变。
