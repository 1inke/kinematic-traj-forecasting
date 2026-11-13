# ARS反方检查2：证据链（同助手 inline，非外部或独立审稿）

结论：PASS_WITH_SCOPE_LIMITS。允许确认注册S0停止，不允许外推所有安全空间失败。

- Major｜最强替代解释：GT逐点球方向包含加速度/jerk改动，逐对应导数1.01原生预算比组内五统计门严格得多。原生导数为零或接近零时，一个全时域alpha可能几乎归零。S0不是原安全域的最优求解，失败不能推出不存在别的安全方向或时间形状。
- Evidence｜FITchange94500候选中47755个最终alpha=0，47109个至少一个原生加速度预算精确为0；alpha99分位9.171589323746326e-7。全部33绑定/零预算频次已保存，含并列，不能把频次当互斥候选比例。
- Counterevidence｜A0放松all20上界仍为3108而native1937，且S0有46745个实际非零修正，不能说all20没有机会或所有路径都bitwise不动。问题限定为本固定方向+逐导数ray的有效步幅。
- Minor｜数值fallback仅47/94500，不是47755个zeroalpha的同义词；不能声称大部分因数值错误被回退。
- Scope｜覆盖不增不等于平均误差绝对不变。FITchange ADE/FDE存在约1e-6百分比量级改善，FIToverall在少数stable候选上变化更大，但不产生新完整合格窗口。
- Audit｜生产独立全路径/聚簇/物理/ray算术PASS max0；补充CPU逐slot指标和certificate/counts/原1863source+8primary+4done SHA初末PASS max0。相同公式的数值重构不是外部科学验证，更不是全部旧模型前向。
- Decision｜保留相反证据后，0>=23625为false依然成立。不能用outer/CAL、敏感阈值、修改margin或训练六次挽救门槛。

