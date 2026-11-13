# 第三步：表示瓶颈与合法观测信息增量

2026-10-26｜DESIGN_ONLY。只有前两步未充分回答问题、且来源审计通过才实施；不自动新增场景地图或未来信息。

## 1. 修正原建议的前提

实际324维特征已经包含past位置、速度、加速度42维，以及最后token和mean token共192维。再“加原始历史”不是新增信息，不能用这种重复特征宣称发现信息缺口。

本步拆开两个不同问题：

1. **表示读取瓶颈**：完整8×96冻结token，比last＋mean摘要是否更容易被当前诊断头利用？它不是新增传感观测，token也未必含有摘要无法重构的任务信息。
2. **观测内容增量**：合法观测期邻居历史，是否提供当前显式输入没有充分表达的增量？原context可能已经编码邻居，因此也不能简单称全新信息来源。

## 2. 四arm、一个共同网络

|arm|原324维|完整8×96 tokens|观测期邻居192维|
|---|---|---|---|
|C_base|有|置零|置零|
|C_token|有|有|置零|
|C_neighbor|有|置零|有|
|C_both|有|有|有|

共同supervector为1284维：324＋768＋192。共同MLP为1284→32→32→1，前两层bias、末层无bias，沿原激活，共42208参数。各arm初始化完整权重配对相同；不用的支路在标准化后严格置零。

这是相同名义结构，不是相同有效容量：置零支路权重没有输入作用。C_token同时改变可利用表示与有效连接，不能把阳性唯一归于信息丢失。当前原324维MLP只作历史参照，主要比较必须在本四arm内完成。

每arm池seed42/43/44，共12拟合；不扫描token数量、邻居数、网络宽度或注意力模块。

## 3. 新特征的严格来源

### 3.1 完整token

只读原保存8×96观测token，按时间正序展平。同一窗口全部401动作共享token；不得把预测期decoder状态、GT监督特征或重算后不同缓存混入。

完整token与原last/mean存在冗余，应保存`token[-1]`、`token.mean(0)`与原192维的数值一致性检查。原缓存若含未来可用性掩码选择，限定为“固定缓存条件下读取诊断”，不能声称完全部署可用；不得静默重造原候选。

### 3.2 过去邻居支路

只使用现有合法原始轨迹中截止t0的8个观测时刻，不能用“未来12步仍存在”作为入选条件。

- 邻居在最后两个观测时刻存在才有资格；按t0距离从近到远取最多8个，tie按原身份稳定排序。
- 每邻居每时刻保存相对目标位置x/y与存在mask：8邻居×8时刻×3=192维。
- 早期缺失位置填0，mask=0；不拿未来点插值，不把0位置误当有效重合。
- 相对位置是同一时刻neighbor−target；单位与dt核对，身份只用于排序、匹配、审计，不入模型。
- 不从已有`num`或有效邻居sidecar直接推断past-only，须追溯生成逻辑。
- 删除或改写审计副本的t0之后数据，邻居身份、mask、所有特征必须完全不变。审计副本只存在新目录，旧数据不改。

若现有文件不能重建真正过去邻居，C_neighbor/C_both标记BLOCKED_INPUT_PROVENANCE，不下载新数据、不使用未来存活掩码替代；C_base/C_token仍可按来源限定完成。

## 4. 固定训练协议

训练ID、全动作T、动作样本及逆采样权重与第二步完全相同，不换rescue-only目标。四arm共用opt-only的1284维统计合同；各块从实际opt特征拟合，不使用hold/inner。二值存在mask和STOP位不标准化，位置/连续特征按共同方差下限规则处理。

AdamW lr=.001、weight_decay=.0001、clip5、batch32窗口、独立平衡BCE、8epochs/1024updates、终态固定。三个seed配对batch顺序，所有arm结束并锁定SHA后才读取hold效果。

```python
def build_supervector(base324, token8x96, neighbors8x8x3, stats, arm):
    # Only observed/cached inputs; no future, group, B/T or oracle action.
    x0 = stats.base.transform(base324)
    xt = stats.tokens.transform(token8x96.reshape(-1))
    xn = stats.neighbors.transform_preserving_masks(neighbors8x8x3)
    if arm not in ("C_token", "C_both"):
        xt = zeros_like(xt)
    if arm not in ("C_neighbor", "C_both"):
        xn = zeros_like(xn)
    return concatenate([x0, xt, xn])
```

建议模块：`past_only_neighbor_audit.py`、`feature_blocks.py`、`run_input_ablation.py`。不新建完整encoder或端到端训练旧F16。

## 5. 主要比较与结论界限

在共同六项家族中，本步三项hold change主要差值为：

- C_token−C_base：完整时序表示的读取增量。
- C_neighbor−C_base：显式过去邻居增量。
- C_both−C_token：完整token之外的邻居增量。

C_both−C_neighbor、全组A/F/J等仍完整报告，但列探索，不替换主要比较。使用共同20k配对track/seed、99.1667%校正CI及原安全/保留门。

|结果|可以支持|不能支持|
|---|---|---|
|token阳性，base弱|当前摘要/小头读取存在改进空间|全部历史信息已丢失、唯一encoder根因|
|邻居阳性且past-only审核通过|现有合法邻居历史提供可利用增量|理解心理意图、地图一定有效|
|只有both阳性|联合表示有用，可能需组合读取|任一单独输入是唯一原因|
|训练改善但hold不改善|表示改动未解决当前迁移|数据没有信息|
|全部阴性|本有限探针未找到增量|输入条件下未来不可预测|

即使信息诊断阳性，只要lost/protected/five-motion或成本门失败，仍不可部署。不得把42208参数实验头直接宣称轻量创新已经完成。

## 6. 预检与保存

真实子集两epoch全状态恢复；实际新分支梯度分阶段有限；inactive block严格0；token与base重叠部分一致；原始future改变不影响新邻居输入；GT改变不影响推理选择以外的构造；实际128窗保存401分数/K20重算。

保存`FEATURE_PROVENANCE.json`、`PAST_ONLY_AUDIT.json`、`TOKEN_REDUNDANCY_CHECK.json`、四mask、统计及输入SHA、12run manifest/终态/history、全决策与完整指标、bootstrap与CPU复核。不能用预检峰值代生产成本。

反方：新输入也改变有效连接、冻结token可能含原掩码选择偏差、邻居有用不代表所有未来可辨识。若第二步已回答同输入可利用问题，本步可明确NOT_NEEDED而非机械执行。
