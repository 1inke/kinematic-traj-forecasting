# DPP双重可行性：阶段A离线诊断

运动学质量与DPP算术完成，场景分支及DSF训练未执行。新训练模型0；没有选择/删除/替换候选。原K20覆盖及minADE/FDE保持不变。
完整逐模型/fit-dev/原C0组与GT转向组分母见ALL_GROUPS.csv。qscene与qjoint均未知，不能称双重可行性已验证。

Scoring tracks against thresholds derived from same FIT distribution is not independent validation. A lower q for high own-penalty tracks is a definition, not evidence of physical safety or generalization.
Fixed paths mean no coverage/minADE/FDE improvement by scoring alone. Valid-only spread and expected cardinality do not establish learned coverage. Missing scene branch is unresolved, not negative. Related windows and same seed42 sources are not independent trials.

经验阈值来自FIT GT99.5%和IQR，不是普适生理边界。评分下限1e-6仅DPP质量数值选项，raw/logq与下溢数均保存；没有dev扫参。
无合格场景图、原DSF接口和真实latent，因此完整B0-B4训练比较未实施。下一阶段须实际补齐这些合同，不能用h3嵌入伪装Gaussian latent或换模型凑实验。

全部文件、失败、旧结果和PAUSED记录保留；同助手ARS反证非外审。
