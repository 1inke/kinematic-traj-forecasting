# 第二步：同输入异构学习器——检验已有信息能否被利用

2026-10-26｜DESIGN_ONLY。执行与统计以[共同合同](00_COMMON_CONTRACT.md)为准。

## 1. 假设与对照

固定候选池、输入来源、T标签与动作样本，比较小MLP和一种树模型。若另一种学习器在隔离track上稳定取得收益，支持“现有输入具有可利用信号，而当前学习方案未充分利用”。这是正证据检验；几种方法都失败不是输入无信息证明。

|arm|输入|模型|目的|
|---|---|---|---|
|M_full|原324维|原11488参数MLP|新的匹配训练基线|
|T_full|原324维|固定Histogram Gradient Boosting分类器|异构学习器探针|
|T_geom|234:324共90维|同配置树|候选几何自身已能解释多少|
|T_perm|324维，但0:234观测bundle错配|同配置树|观测与动作对应关系负对照|

每arm使用池seed42/43/44，共12次拟合。不是同时比较多种树、核、网络宽度或训练预算。geometry-only不是无信息：原native/KIN已经编码历史。

## 2. 数据与训练固定

FIT_opt4096训练、FIT_hold3825主要诊断、inner1856次要。整track身份沿用，不重新随机窗口划分。旧I/S结果不能充当本轮匹配基线，因为本轮使用共同合同第4节的分层动作样本。

各seed最多131072训练动作行；四arm共用同一行ID、原T及原权重。无负类窗口不训练，但留在所有评价分母。评价始终all401，不只在训练抽中的动作中选择。

MLP固定原初始化/激活、独立平衡BCE，AdamW .001/.0001、clip5、batch32窗口、8epochs，即1024updates；终态固定，不用hold选模型。批loss先按窗口累加共同权重再对窗口平均。

树配置固定如下，直接使用共同合同原始sample_weight，不额外重平衡类别、不根据效果更改权重尺度：

```python
HistGradientBoostingClassifier(
    loss="log_loss", learning_rate=0.05, max_iter=64,
    max_leaf_nodes=8, max_depth=3, min_samples_leaf=20,
    l2_regularization=0.1, max_bins=64,
    early_stopping=False, categorical_features=None,
    random_state=pool_seed,
)
```

API依据：[scikit-learn官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)。执行前记录本机版本并核实支持这些参数；本设计不宣称本机版本、不授权安装或升级。树容量与MLP不等价；本步是学习器家族诊断，不是纯优化器因果对照或轻量部署方案。

## 3. 负对照如何构造

不能仅在测试时打乱输入，把OOD下降说成“输入有信息”。T_perm必须从训练起使用错误对应关系，hold/inner也按同样规则构造。

1. 用FIT_opt最后观测速度拟合三分位切点，按`sequence × speed_bin`分层；重复切点可形成空层，原样记录，不改切点搜索。
2. 各split内按SHA256(`info-observation-permute-v1|split|pool_seed|ID`)排序。
3. 在每层寻找最小正循环位移，使每个recipient与donor的target track不同。只使用身份和过去速度，不使用未来或动作标签。
4. 替换整个0:234观测bundle，保持bundle内部历史/context一致；recipient候选几何、eligible、标签、ID不变。每个窗口全部动作共享同一donor。
5. 若任何非空层找不到合法位移，T_perm标记BLOCKED，不合并层、删样本或修改规则求成功。其他三个arm可以完成，但不能虚报完整负对照结论。

该错配仍可能破坏观测与候选的正常联合分布。因此它只是错误对应关系控制，不是严格条件独立检验。T_full优于T_perm，仍须同时优于T_geom与native，且承认该限制。

## 4. 核心实现接口

```python
rows = build_shared_action_table(opt, tag="info-probe-v1")
assert rows.matches_all_arms()
X, y, w = rows.features, rows.is_in_T, rows.class_ipw
tree.fit(X, y, sample_weight=w)

# inference: no GT, no B/T, no sampled-action restriction
scores = tree.decision_function(all_401_features)
scores[~eligible] = -float("inf")
action = stable_argmax(scores, tie_order="STOP_then_action_id")
paths20 = apply_one_edit(native20, kin20, action)
```

分数只是决策分数，不解释为未来发生概率。实施模块建议：`shared_action_table.py`、`observation_permutation.py`、`fit_probe.py`、`evaluate_401.py`。全12个预定终态先锁SHA，再一次解锁hold效果；禁止看前三个结果后改剩余参数。

## 5. 指标和分支判定

主要差值：hold change的T_full−M_full、T_full−T_geom、T_full−T_perm；共同六项家族校正，不另缩小比较分母。覆盖/保留/五motion、三seed差异、训练与hold的hit_B及rescue利用率全部报告。

- **学习方案不足正证据**：T_full−M_full达到共同正信号门，且T_full自身相对native也过门。仅比一个退化MLP好不算。
- **观测增量正证据**：T_full−T_geom及−T_perm都达门，且自身优于native。只允许称该固定输入/人口下存在增量预测信号。
- **几何已足够有用**：T_geom相对native有收益但full未增加；列为探索结果，不能强称历史无用，历史可能已通过生成器进入几何。
- **全部弱或CI跨0**：仍不确定；进入第三步有限输入对照，不追加模型家族扫描。
- **训练内好、hold弱**：支持当前映射迁移不足，不能唯一归因共享head干扰。

若已获得清楚的同输入正证据，可结束“有没有可利用信息”这一诊断目标，把第三/四步标记NOT_NEEDED并交付；不为凑满系列跑全部模型。

## 6. 预检、保存与局限

除共同预检外，逐行核逆采样权重、STOP强制保留、无负类处理；验证训练subset不会泄入推理候选；扰动GT不改变已训练模型的推理特征和eligible。树若不支持严格中途恢复，明确声明完成SHA-skip，不伪造optimizer/RNG恢复。

保存12个manifest/model、共同ACTION_TABLE、PERMUTATION_MAP、全401分数与选择、训练/hold/inner完整指标、全部bootstrap抽样与独立CPU复核。旧K1树回归器、旧R4近邻不是本轮多正例单编辑分类器的同协议重复，应列入复用矩阵但不冒用其结论。

反方：①树阳性可能来自容量和结构，不独证优化器差；②错配负对照存在联合分布破坏；③留出已反复开发，只是探索性新头迁移。不能据此宣布理解行人意图或部署成本合格。
