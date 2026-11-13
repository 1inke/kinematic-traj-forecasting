# 互补蒸馏 E0/E1 后续处理实施指南

日期：2026-10-22  
状态：**指导文档，尚未执行本指南中的修正、补评或新训练。**  
对应建议：“先修正报告与来源记录”，随后冻结补评，再决定是否值得做新的受控实验。

## 1. 目标与边界

要回答三个问题：

1. 旧 E0/E1 的哪些结论可引用，哪些必须降级或纠正？
2. 现有学生是在训练集内就没有学好，还是训练内改善而迁移失败？
3. 排除教师成员、信息输入和损失尺度混杂后，还有哪个具体机制值得测试？

本轮优先做证据修复，不直接容量翻倍、过采样、增加 epoch 或修改 scorer。蒸馏是实现互补的一种训练途径，不是互补机制的必要定义。

执行分层：

|阶段|工作|需要训练吗|完成条件|
|---|---|---|---|
|P0|冻结现有证据、建立文件清单|否|来源与缺口全部可定位|
|P1|修正报告、门状态、统计和上游暴露表述|否|逐条修订有证据，不覆盖历史|
|P2|用已有 11 个 checkpoint 冻结补评 FIT/inner|否，仅前向|22 个模型×人口输出及独立保存算术核验|
|P3|根据 P2 选择下一条研究分支|否|明确继续、停止或来源受限|
|P4|仅在 P3 支持后，另行冻结一个新单因素协议|可能|不是本指南自动启动的任务|

本指南不启动 worker、不修改旧 checkpoint、不补造 optimizer/RNG，不恢复旧暂停任务。保留短预算偏好：若单阶段实测规划超过 30 分钟，先交付已完成证据和剩余成本，不默认启动长任务。进程 6 GiB 上限、系统可用物理与提交余量各至少 6 GiB 的既有限制保持。

## 2. 必须先读的证据

以下路径均相对项目根 `<WORKSPACE_ROOT>`：

|材料|路径|用途|
|---|---|---|
|原总报告|`qoder_work/互补蒸馏_E0E1_实验分析报告_20261020.md`|待修正的叙述，不作唯一事实源|
|已有复核|`qoder_work/E0E1_REVIEW_20261020/ANALYSIS.md`|逐条问题、已核验边界|
|保存复算|`qoder_work/E0E1_REVIEW_20261020/SAVED_REVIEW.json`|E0 覆盖、教师可教比例、关联 CI|
|原 E0|`qoder_work/E0_CALV11_ORACLE_OPPORTUNITY_20261020/`|RESULT、实际 kin_src、保存选择索引|
|原 E1|`qoder_work/E1_STUDENT_DISTILL_SCREENING_20261020/`|DESIGN、三份 RESULT、11 个学生权重、当前执行源码|
|划分|`qoder_work/v1_1/SPLIT_V1_1.json` 及索引、身份 CSV|核验人口与索引映射|
|学生实现|`HYBRID_NATIVE_KIN_UNION_20261018/kdec/models.py`|实际导入的 BASE 学生，不采用同名其他目录版本|
|数据与评价|`CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k0/`|实际 loader、坐标、ID、评价函数|

另定位 `TRACK_OVERLAP_AUDIT_20261020` 的实际材料，追溯每一份 native/KIN 缓存使用的上游 checkpoint、训练身份与预处理统计。不要用某个名称相同的 checkpoint 替代实际来源。

本指南的历史数值以 2026-10-20 复核为依据；执行时若文件已变化，应先比较版本，不能静默套用旧清单。当前源码可用于重放接口核对，不能因此宣称它就是每个旧 run 的训练时源码。

## 3. P0：保全原件，建立可追溯来源链

### 3.1 新建修复目录

建议新目录 `qoder_work/E0E1_EVIDENCE_REPAIR_20261022/`。本指南没有创建该执行目录。将所有新脚本、报告、输出放入其中，旧目录只读。

建议产物：

```text
E0E1_EVIDENCE_REPAIR_20261022/
  README.md
  protocol.json
  source_manifest_before.json
  source_manifest_after.json
  RUN_INVENTORY.csv
  CACHE_LINEAGE.json
  EXPOSURE_AUDIT.csv
  CORRECTIONS.csv
  REPORT_REVISED.md
  frozen_eval/
    predictions/             # 一个 checkpoint × 一个 split 一份
    sample_metrics/
    metrics_by_group.csv
    assignment_stats.json
    checkpoint_replay_manifest.json
  completion_review/
    REVIEW.json
  DECISION.md
```

这些是拟产物，不是已完成文件。首次实现先做清单与元数据，不批量加载全部模型或预测数组。

### 3.2 记录到实际读取文件，而不只是“同名缓存”

每个输入至少保存：绝对路径、角色、大小、当前文件 SHA256、数组 shape/dtype、显式 sample_id 或可验证的顺序映射、坐标和时间单位、生产者 checkpoint 路径/SHA、训练及统计拟合身份来源、证据等级。

特别检查 E0：实际 `kin_src` 指向 `COMPLEMENTARY_POOL_VALIDATION_20261018/artifacts/v1/`；旧输入清单却列 `UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/`。必须分别记录，不能以后一位置的文件哈希替代前一位置。

文件哈希与数组哈希用途不同：前者验证整个文件，后者验证解析后的内容。若缓存没有 IDs，必须证明生产顺序与注册表顺序一致；仅行数相等不能通过。

建议来源字段：

```json
{
  "artifact_path": "待填写实际绝对路径",
  "file_sha256_now": "待计算，不能填占位值为PASS",
  "producer_checkpoint_sha256": null,
  "producer_training_ids_sha256": null,
  "producer_statistics_ids_sha256": null,
  "historical_source_binding": "UNKNOWN",
  "evidence_kind": "POSTHOC_RECONSTRUCTION",
  "replay_verified": false,
  "upstream_exposure_status": "UNKNOWN"
}
```

证据等级分为 `CONTEMPORANEOUS_RECORD`（原运行同期记录）、`POSTHOC_RECONSTRUCTION`（现在追溯重建）、`UNKNOWN`。现在补清单不能将第二类升级成第一类。

### 3.3 分开审核下游划分与上游暴露

对 native、每个 seed 的 KIN、学生各自建立训练/验证/统计拟合身份链。至少核验 sample、sequence+track、时间区间交叠；缺身份或时间时标未知，不当作无重叠。

- v1.1 的 FIT/inner/outer 与 CAL 身份时间隔离，是当前划分层面的结论。
- 缓存生产模型是否曾训练或选择于相同评估人口，是另一层结论。
- 上游来源存在暴露或无法核实：允许继续描述性重放与修正报告，但将相应结果标为探索，不能宣称 whole-model 未见泛化。
- 如果连当前预测与 GT/ID 都不能唯一匹配，停止该人口数值分析，报告具体缺口。

不为了“变干净”而偷偷换划分或重新生成缓存；那会成为另一项实验。

## 4. P1：报告修订清单

保留原总报告；新建 `REPORT_REVISED.md`，开头注明取代哪些叙述、原报告路径与哈希。`CORRECTIONS.csv` 使用 `claim_id, original_claim, corrected_claim, evidence_path, verification_level, unresolved` 字段。

必须处理以下条目：

|旧表述或风险|修正口径|
|---|---|
|“干净的 +8pp 上界已证明”|既有有限候选池在 cal_v11 上存在 GT 辅助可达见证；上游暴露尚不能由新划分自动排除|
|K0-C 是数学上界|是 GT 贪心可行解，没有最优性证明，也不约束所有可能学生|
|clean/dropped 增益相近说明污染无影响|两人口组成不同，没有随机化或匹配暴露对照，不能作因果结论|
|原三 seed CI 直接引用|注明共同 track 关联的事后敏感性修正；旧 CI 保留为历史，不继续作为主联合区间|
|seed42 筛选全部通过|原 JSON 中运动门失败，三 seed 追加属于失败后的探索补跑，不是合规晋级|
|“学生容量不足已证实”|当前 history-only、57,432 参数、8 epoch 配置未可靠成功；信息、优化、预算与容量尚未分离|
|26% 失败窗可教|区分“至少一条好教师的窗口比例”与“平均好路径条数”|
|B/E 只改变是否分解|教师成员和归一化均不同，不能单因素归因|
|输入全锁源、可完整续训|仅已证实的文件范围；旧权重未绑定完整 optimizer/RNG/训练源，不能补造可续训事实|
|旧 FATAL 代表所有最新训练失败|按来源与时间区分早期索引失败和后续已有结果，不删旧失败|

### 4.1 可使用的数值与证据层级

E0 保存索引的覆盖已由已有 NumPy 复核重建：

|人口 change|原生|seed42/43/44 贪心集合|lost|
|---|---:|---|---|
|cal_v11，443 窗|146|182 / 180 / 186|0 / 0 / 0|
|dropped，257 窗|97|117 / 117 / 116|0 / 0 / 0|

cal_v11 平均增量 8.2769 个百分点。已有事后共同 track 抽样、固定三 seed 区间为 [5.8411,10.9568]pp；共同 track 加 paired-seed 重抽敏感性为 [5.5556,11.2710]pp。保留原复核的 20,000 次抽样与 RNG 设置，不反复换 seed 求窄区间。同一 draw 必须给全部模型使用相同 track 重复次数；窗口加权时用抽中的窗口数作分母，不能无说明改成 track 等权。

这是事后探索区间，不是新确认性检验，也不证明新场景泛化。已有复核未全量重算 E0 的 W、五运动学和聚簇；修订报告必须把这些标成原执行器报告，或另列完整重算为未完成项，不能跟覆盖复算混在一起写“全部独立通过”。

E1 原 JSON：A/B/E change 分别为 140/145/161、137/133/129、148/137/138；E overall 为 700/641/646，均低于 B 的 785/659/665；native overall 932/1856，change 166/497。11 runs×4 groups 的原 motion_gate 都是 FAIL。此处是 JSON/源码核对，不冒称 E1 保存路径已独立审核。

FIT 的 8051 个 native 失败窗中，可教窗为 1943/1952/1899，即约 24.13%/24.25%/23.59%；change 失败 2788 窗中的可教窗为 448/481/483。不能用路径条数代替窗口数。

## 5. P2：已有 checkpoint 冻结补评，不训练

### 5.1 固定范围

完整清单是 A/B/C/D/E×seed42，加 A/B/E×seed43、44，共 **11 个已存在模型**。每个模型评估原 FIT 18909 与 inner 1856，共 **22 个 checkpoint×split 输出**。不是补跑缺失的 C/D seed43、44，不新增 outer 或 CAL 消耗。

固定最终已有 checkpoint，不从结果挑最好 epoch。每次只加载一个模型，`eval()`、禁梯度，不创建 optimizer、不 backward、不更新参数和统计。CPU 或 GPU 冻结前向均可，但记录设备与 dtype；先小样本确定数值重放口径，不能事后放宽差异阈值以复现想要的结果。

新重放报告必须称“当前已核对实现加载旧权重的冻结补评”。除非有同期源码证据，否则不能把它称为原训练协议完全重建。

### 5.2 实施顺序

1. 先静态检查实际导入路径和顶层副作用，禁止直接调用旧 `run_e1.py` 的 main。
2. 从实际 `M.build("BASE")` 的对应类构造学生，严格匹配 state_dict；漏键、多键、shape 不符直接停止，不用 `strict=False` 隐藏问题。
3. 核对原 loader 的 past/GT/native 坐标系、FIT-local 与全局索引，保存显式 ID；不能通过数组长度猜对应。
4. 哈希选择固定小样本，只按 ID 选，不按质量挑；先验证有限值、K20×12×2、坐标、批次切分一致性和权重/全部 buffer 前后不变。
5. 完整重放 inner，比较原 JSON 的 N、native 数、good/rescue/lost、误差。离散计数若不同，逐窗解释边界或重放版本差异，不覆盖原结果。差异解释不清时标 `REPLAY_NOT_RECONCILED`，不继续训练建议。
6. 再补 FIT，保存全路径后由另一个纯 NumPy/SciPy 计算入口复算指标。该独立算术入口仍是同一助手编写，不称外审。
7. 每个模型处理完释放对象；失败保留新日志和部分产物，不能拿旧源错误当成当前 checkpoint 全部不可用。

### 5.3 必须保存什么

每个模型×人口的 NPZ 至少保存 sample_id、sequence、track、group、past 最后点、GT、student20、native20；教师可用人口另存教师来源、mask 与精确行映射。记录文件/数组 SHA、shape/dtype、模型 SHA、统计 SHA、加载源码 SHA、运行版本。

评价不拼接候选：

```python
# 设计伪代码，不是已实施脚本
err = norm(pred - gt[:, None], axis=-1)       # [N,K,12]
ade, fde = err.mean(-1), err[..., -1]
good = (ade <= 0.2) & (fde <= 0.4)
covered = good.any(axis=1)
j = maximum(ade / 0.2, fde / 0.4)
j_slot = argmin(j, axis=1)                   # 同一条完整轨迹
rescue = (~native_covered) & covered
lost = native_covered & (~covered)
assert covered.sum() - native_covered.sum() == rescue.sum() - lost.sum()
```

分开报告下列指标，不能只看 change 最好的一项：

- overall/stable/transition/change 的 N、coverage、rescue、lost、retained、minADE、minFDE、J；minADE/minFDE 可来自不同槽，不能拿它们拼完整成功。
- 三个原生参考子集：native 已成功；native 失败但存在好 KIN 教师；native 失败且没有好 KIN 教师。补全率的分母为第二类 N，lost 率的分母为第一类 N；空分母标 NA。
- 对 inner 若没有可验证同 ID 的 KIN 缓存，仅报告 native 成败子集，教师补全集标 unavailable；不要临时训练教师或借用 FIT 行。
- 每个槽的 WTA 责任数、教师最近匹配次数、至少一个成功候选的槽分布、tie 数；分别统计“赢过 GT”与“被教师选过”，不能合称使用率。
- 原 GT-free complete-link：meanpath .10m、endpoint .20m 的既有距离规则与完整 K20，报告 distinct/effective-good；不拿候选更多或簇更多等同质量。
- 全组 pooled speed p95、acceleration mean/p95、jerk mean/p95，dt=.4，沿用原起点、差分和 quantile 实现；逐项与 native 比较，不用逐窗 p95 的均值代替全人口 p95。
- 用固定 J 最近的同一个槽计算前/中/后 4 步误差，不分段另选 oracle。

原五运动学 ≤native×1.01 的标准保持；旧 E1 的筛选门与项目更强 major 门分开列，不事后替换旧协议。native 基值为零时直接比绝对不等式，百分比记 NA，禁止加评价 epsilon。

当前仅有最终权重与总 loss history：可以补最终 FIT/inner 指标及冻结损失分项，**不能补造历 epoch coverage、历史槽责任或优化器状态**。没有存过的训练轨迹明确 unavailable。

### 5.4 输出验收

- 11 个 checkpoint 和 22 个人口输出清单闭合；缺失项明确说明，不替换成另一个模型。
- 每份输出 ID 唯一且与输入逐行一致；全 20 候选有限。
- 所有旧输入、权重前后 SHA 一致；冻结前向参数与 buffer 不变。
- coverage/rescue/lost 恒等式通过，完整候选成功不混 minADE/minFDE。
- 保存路径重算与直接计算一致；离散指标要求一致，浮点容差按算法/精度事前记录。
- 重放与旧 JSON 的差异有逐项说明。通过保存算术不等于历史训练源绑定问题已经消失。

## 6. P3：看完补评后怎样分支

|观察|允许的推断|下一步|
|---|---|---|
|FIT 和 inner 都差，多个槽长期同质或几乎不被使用|当前学习配置/匹配可能有问题；不独证容量不足|先有限的实现、输入、匹配诊断；不立即加宽并过采样|
|FIT 明显改善，inner 无改善或 lost 明显增加|迁移与保留问题优先|保留同模型与预算，设计单一保留/补全权重实验；不把训练拟合当泛化|
|补全子集变好，但 native 已成功子集显著变差|任务权衡有证据|同教师集合、总权重固定，单独比较组间权重分配|
|history-only 各臂整体弱，但 F16 信息未提供给学生|信息与容量仍混杂|把“F16 内互补生成”与“history-only 知识压缩”分开研究，不互相替代结论|
|质量有局部收益，但五运动学仍失败|尚不能部署/升级 baseline|先定位教师与学生安全失配，不事后过滤坏结果声称方法成功|
|来源未解决而点估计好|存在探索信号，不是无污染泛化|完成 lineage 或另立干净模型实验；不继续堆显著性|

本阶段只形成 `DECISION.md`：证据、替代解释、推荐唯一因素、停止理由、成本边界。没有支持继续的信号也可以结束，不必须凑下一轮训练。

## 7. P4：若继续，如何做真正可解释的蒸馏对照

### 7.1 先选研究问题，不混两条主线

- **F16 同信息互补生成**：全部实验臂共享同一可核实来源的冻结 F16 表征、同一 K20 学生结构、初始化与数据预算。回答同信息下集合监督有无增量。
- **history-only 轻量压缩**：全部臂保持当前历史输入，研究低成本学生能保留多少教师覆盖。不能用其失败否定 F16 表征下的生成能力。

根据 P2 只选一个；如果新引入表征，所有臂共同使用，并以这个新公共基线做内部比较，不能与旧 history-only E1 直接作机制归因。新表征来源若不满足评估边界，仍只能探索。

### 7.2 固定教师全集和目标分组

推荐先采用旧 E 实际使用的集合：R=原生成功窗中的原生好路径；N=原生失败窗中的 KIN 好路径。T=R∪N，两组不相交。native 成功窗内的 KIN 好路径本轮对所有臂一致排除，排除数量必须记录。也可另定完整 union，但必须在训练前选择一次，所有臂共用，不能看到结果后更换。

教师质量阈值不是集合运动安全证书。若新增教师物理筛选，必须另立因素；不能在本轮一边变权重、一边悄悄换教师。

### 7.3 先做代数等价单元测试，再谈科学实验

设 d(t,S)=min_s[mean_timestep||t-s||+.5||t_end-s_end||]。某 batch 两组距离和为 S_R、S_N，有效教师数为 n_R、n_N。

普通蒸馏：

`L_union = (S_R + S_N) / max(n_R+n_N,1)`。

等价分解：

`L_decomposed = n_R/(n_R+n_N) * S_R/n_R + n_N/(n_R+n_N) * S_N/n_N`，空组项为零，空 batch 总项为零。

**两者数学相同。** 同 teacher、同逐项权重、同学生匹配时，改成两个函数名不会产生新机制。必须验证标量与全部梯度一致；实现差异只能按数值/代码问题处理，不作为创新收益。无需为这个恒等式花九次训练预算。

### 7.4 一个可辨识的候选实验：固定总预算的组间权重

若 P2 显示保留/补全权衡，正式比较限定为三个臂×三配对 seed：

|臂|目标|作用|
|---|---|---|
|A|原基础损失|新公共基线|
|B|基础损失 + λ L_union|同集合普通蒸馏|
|E_bal|基础损失 + λ L_bal|仅改变 R/N 的相对权重|

固定 λ=1，基础损失、梯度裁剪、优化器、初始化、batch 顺序和预算全部使用同一已核实配方；不能混入其他系列的 noclip/25 epoch 配方。若沿用原 E1，则三臂共同固定 8 epoch、batch256、AdamW lr=.001/wd=.0001、grad clip5、原 diversity .05，最终 epoch primary。它只是短预算筛选，不宣称已收敛。

`L_bal` 规则事前固定：两组均非空时，各自教师均值权重 .5/.5；只有一组非空时该组权重 1；都空时为 0。全局 λ 不翻倍。逐 batch 保存 n_R/n_N、有效权重与损失分项，不改变采样顺序来强行塞入两组。

这检验的是**组间重加权**，不是“分解命名本身”；梯度幅度、方向与方差可随权重变化，不能声称相同 λ 就梯度强度完全一致。若需要控制梯度强度，那是另一个精确设计，不能临时加入。

B/E_bal 共用信息、结构与教师成员；不得同时加宽、过采样、换匹配距离或加物理项。新 checkpoint 必须绑定完整源码、IDs/统计、teacher masks、objective、init、optimizer、全部 RNG/history，且真实恢复测试通过后才训练。

primary 是 inner 上 E_bal−B 的完整覆盖/保留与安全；A 用于判断普通监督是否有用。三个 seed 完整报告，不能单 seed 好就当确认。后续 outer/CAL 只在新协议明确其探索性质和使用次序后使用。具体 CI 家族、效应阈值和停止门在正式启动前补成冻结 protocol；本节不是已经可直接启动的完整训练协议。

## 8. 交付与停止标准

修复包完成不要求实验转正。最少交付：

1. 修订报告与逐条纠错表；原报告不覆盖。
2. 当前输入/来源清单及上游暴露表；未解决来源保留 UNKNOWN。
3. 11 模型 FIT/inner 冻结补评，或具体缺失/阻碍清单。
4. 保存结果独立算术审核与原 JSON 重放差异说明。
5. 一页决策：是否继续、唯一新因素是什么、什么证据会否定它。

直接停止数值推进的情况：ID 无法对齐、坐标无法复原、严格权重加载失败、当前源/输入变化、资源门不满足。历史 provenance 不完整可允许限定的描述性补评，但阻止无污染泛化主张；两者不要混成“全部不可研究”或“全部通过”。

没有优化器/RNG的旧权重只用于冻结评价，不能称 exact resume。routine reader/精度问题由实现者排查，保留失败来源，不通过放宽科学阈值解决。

## 9. 设计反方检查（同助手，不是外审）

- 范围检查：来源修复不能变成偷偷重跑整条上游训练。指南将重放、历史追溯与新实验分开。
- 证据检查：+8pp 仍可能受上游暴露、人口组成和 oracle 选择影响；保留见证，不推出可部署收益。学生失败也不能证明互补无效。
- 方法检查：完全同权重的“分解”是恒等式；真正的组间重加权会改变优化几何，必须如实命名并保持其他因素一致。
- 最强反驳：即使证据修复和内层改善完成，仍未证明未知场景、独立整模泛化或运动理解。论文主张必须停留在实际完成的边界。

本指南由 AI 助手依据本地报告、已存复核和实际 E1 入口源码整理；本轮未运行 checkpoint、bootstrap、完整 motion 审核或新训练。研究设计技能用于范围、证据等级与反方检查，不代表外部独立认证。
