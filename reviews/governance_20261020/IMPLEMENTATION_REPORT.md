# 两份文档的工程落地与当前边界（2026-10-20）

依据：作者提供的 `PROTOCOL.md` 与 `DESIGN_AB.md`。根目录保存新副本，原下载文件未修改。当前为公共工程实现、S1组件测试、既有保存数据审计，**0次正式训练、0 GPU训练小时**；不是A/B正式结果或作者S0冻结。

## 已实现

1. `src/f16/`：显式ID连接和冻结SHA、visible-only坐标变换、FIT-only统计、唯一完整候选good、Q-winner同路径ADE/FDE、J/分段、GT-free complete-link、12v/11a/10j五pooledmotion、track→seed配对bootstrap、Holm p调整、G1–G5测量、四类判定及缺护栏fail-closed。
2. `audit/verify.py`：不导入训练评价，直接独立计算保存路径全部字段；另核native/GT/group/track与冻结源的ID和SHA，篡改GT/track/结果会拒绝。JSON记录脚本SHA、字段数、逐项差异、实际计算的状态。`audit/statistics_verify.py`用另一种逐track展开实现核对完整bootstrap draws。
3. `governance.py` / `checkpoint.py` / `provenance.py`：未作者冻结禁止S2以上、CAL仅S4签字、outer一次消费账本、新运行全部optimizer/RNG与完成文件SHA合同。CPU微型连续—中断exact测试通过；**不是A/B真实GPU/完整FIT训练预检已通过**。
4. A：other-track histogram、同recording causal eligibility、support fallback、top4模式组件。B：sigmoid soft双head混合、J平滑覆盖、批量usage熵、FIT-only teacher Chamfer。所有公式不明的选项要求显式传入，不假装已冻结。
5. 路径配置、pytest/CI、安全扫描、环境记录、任务1.3倍预计峰值/剩余内存及Windows保留峰值规则。邮件兼容入口已改为仅ENV读取；旧邮件JSON留原位且不再读取，未发邮件/未改监控。
6. README / BASELINE / docs、287个历史目录索引。旧目录**未整体移动**，因为原脚本及历史SHA引用依赖现位置；迁移风险与后续计划见 `archive/MIGRATION_PLAN.md`。旧根README有归档文本快照；根README、.gitignore、email_notifier是明确授权的治理改动，来源差异另记，不能宣称这三文件旧SHA仍相同。

## 实测结果

- 具体pytest数量、耗时和实际环境见 `pytest.xml` 与 `VERIFICATION.json`；包含T1–T8、Q-winner而非minADE的反例、20copy单簇、同candidate门、独立审计坏输出/GT/track拒绝、bootstrap draw顺序、G5整模预算、缺护栏不PASS、CPU checkpoint optimizer/RNG、ENV邮件凭据测试。
- D-0：native+旧EXP03六套完整outer输出共7套独立几何/聚簇/运动审计；6份旧报告共同字段回放通过，新增Q-winner字段不假造旧值。三个contrast全部20k draws由第二实现复核。文件及旧checkpoint/history前后SHA不变。详见 `experiments/2026-10-20_D0_protocol_calibration/RESULT.json`。
- 近似80% MDE：A−native6.823pp、B−native4.582pp、B−A5.738pp。为双侧α=.05正态近似和本contrast噪声估计，不是统一功效保证或新阳性结果；作者须决定冻结目标。

## 会阻止正式实验的问题

1. **C3**：当前FIT与CAL29个共享track；CAL457窗（change130）与FIT20索引区间相交。CAL295窗（change86）的未来部分与FIT交叠；完整12未来点落入单个FIT区间为0，两个口径分别报告。按原协议先提出v1.1 track互斥方案，再经作者审阅；本轮不偷偷改ID。
2. **C6**：上游h3/v2实际训练split涵盖outer全部1,035窗，1,031窗已知时间交叠、4窗未知。这不是whole-model未见评价。只重新划newhead训练无法洗掉底座暴露。KIN每checkpoint训练IDs仍缺完整manifest；HGB实存rows与CAL也有333窗交叠。
3. 文档未给出精确Holm同时区间构造、G3/G4跨seed聚合、A0具体prior超参数/causal fallback作用域、B软WTA/softmin具体选择/temperature及权重schedule、native/KIN原生adapter/整模305k预算。均列在 `config/*draft.json`；不得以助手默认值冒充作者冻结。

## 尚未完成，不能写成已完成

A的raw-world片段/保留时间purge、SHUF/XSCENE、A0 NLL与top4机会门、query注入；B原生共享encoder/native/KIN adapter、teacher asset、B0–B3训练runner；A/B各自实际GPU、全参数/缓存/源和真实子集2epoch全RNG预检。现有测试只证明新公共组件的指定行为，不证明每个旧runner自动采用了新协议。保留旧科学代码而不原地改变loss/数据/评价。

## 反方审查与限制

同助手使用academic-research-suite架构/DA检查与nature-statistics审查，不是三个独立AI或外部审稿。

- DA1：不能把“shared track为0”当整个模型无泄漏；单独保留原生模型训练暴露、未知时间、邻居context与CAL历史接触。
- DA2：不能将普通95%百分位CI改名Holm校正区间，不按window伪重复；未选作者MDE，不能挑最有利contrast替项目定功效。
- DA3：独立审计不得只相信NPZ自带GT或脚本写死PASS；实际对照冻结源、逐字段比较、坏输出/GT/track测试。源码独立实现不等于外部独立研究者复核；作者关闭审计仍待完成。

## 下一步（严格按文档门）

先提交v1.1分区和底座暴露处置供作者审阅，结合D-0明确MDE、统计和A/B缺失定义；随后才可锁新S0并实现A0/B的actual adapter及S1真实预检。当前不安排后台训练、S3外层调参、S4解封、baseline晋升或自动监控；历史cycle11与原goal不重置。
