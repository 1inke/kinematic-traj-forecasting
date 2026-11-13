# 第一步：小样本拟合与优化可达性

2026-10-26｜DESIGN_ONLY；没有启动训练。共同规则见[00_COMMON_CONTRACT.md](00_COMMON_CONTRACT.md)。

## 1. 要回答的问题

在候选、动作标签和当前324维输入完全不变时，当前小MLP是否至少能够拟合一个固定小训练集？

本步区分“当前优化设置连小样本都难以拟合”与“可以拟合，但大样本学习/迁移仍有问题”。**成功不证明输入具有可泛化信息；失败也不证明输入没有信息。**

最新E1中，S在FIT变化组只获得2/0/1个净覆盖增益，远低于同分区oracle的93/105/102。这使训练内辨识检查有必要，但不是继续加长全量训练的理由。

## 2. 固定实验单元

|项目|预注册值|
|---|---|
|候选池|只用pool seed42的既有native20/KIN20|
|窗口|FIT_opt中按SHA256(`info-tiny-v1|decimal_ID`)排序前128窗，不看GT挑选|
|动作|每窗全部401动作，原eligible及T标签不变|
|输入/统计|原324维、原全FIT_opt统计；不在tiny上重估|
|网络|原324→32→32→1，末层无bias，共11488参数；激活/初始化精确继承原editor|
|训练目标|I原独立平衡BCE；S原多正例ranking；各自直接复用锁源公式|
|初始化|42/43/44，I/S配对；注意这不是三个候选生成器seed|
|优化|AdamW lr=.001、weight_decay=.0001、clip_grad_norm=5；batch32|
|预算|每模型恰好512 updates，共6模型3072 updates|
|读数|update0/32/128/512；只有终态512用于主要判定|

不搜索宽度、学习率、训练步数、margin；不按tiny结果重新选128窗。任何历史小FIT实验先核REUSE_MATRIX，只有输入、标签、动作、损失、样本完全匹配才替代本步。

## 3. 实施顺序

1. 只读绑定128 ID、全401 eligible/B/T、324特征及原统计SHA。
2. 统计B非空/空、rescue机会、稳定/过渡/变化分母。B非空和B空各不足20时标记DATA_SUPPORT_LIMITED，不更换样本。
3. 检查完全相同的模型输入行是否对应冲突标签。按实际float32字节分组，不用人为距离阈值。逐行BCE标签矛盾与窗口top1是否能选中T分别报告，不能混为一谈。
4. 做损失与mask夹具、首层梯度分阶段检查，以及共同合同要求的真实两epoch精确恢复。
5. 完成全部6个固定训练，保存四个固定读数和终态；本步不访问hold/inner效果。

零末层的第一个update前层梯度为0是正常初始化行为，不应因此改网络。若出现非有限数、标签错位、统计泄漏或恢复失败，停止并保留来源，不把工程错误当科学阴性。

## 4. 核心代码设计（接口，不是已实现程序）

```python
ids = hash_order(opt_ids, tag="info-tiny-v1")[:128]
cache = load_sha_bound_rows(ids, pool_seed=42)  # all 401 actions
for init_seed in (42, 43, 44):
    for arm in ("I", "S"):
        model = fresh_original_editor(init_seed)
        opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = exact_seeded_window_loader(cache, batch_size=32)
        for update in range(1, 513):
            batch = loader.next_cycling_batch()
            scores = model(batch.features).squeeze(-1)
            loss = original_locked_loss(arm, scores, batch.eligible, batch.T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if update in (32, 128, 512):
                save_all_action_diagnostics(model, cache, update)
```

`original_locked_loss`必须来自既有I/S实际源码，不能按文字重新猜公式。选动作只按eligible分数argmax，tie按原动作顺序STOP优先；GT只用于训练与事后指标。

建议实现文件：`tiny_manifest.py`、`run_tiny.py`、`audit_tiny.py`；全部写入未来新execution子目录，不改原editor。

## 5. 判定与后续

每个arm独立判定，三个初始化均满足以下条件才称“小样本决策拟合通过”：

- 选中T的窗口比例≥95%。
- B非空窗中选中B比例≥90%。
- B空窗正确STOP比例≥90%。

这些是本次诊断的操作门，不是信息论阈值或项目major门。任一条件人口不足20则不宣称通过。

另外完整报告：loss曲线、错误窗口、每类B基率、真rescue利用率、最终coverage与lost、同candidate J、五motion。B中包含未跨合格阈值的微小改善，因此高hit_B不必然获得oracle覆盖收益。

|结果|允许的结论|下一步|
|---|---|---|
|I/S至少一个通过|当前结构能拟合有限样本，不能称已获得可迁移规律|进入同输入留出比较|
|I通过S失败或反之|目标与优化存在具体差异线索|记录差异，不现场改loss；仍用第二步固定方案|
|两者都失败，数值/来源均正确|当前表示、容量或优化存在瓶颈，尚不能定位唯一原因|只做第二步已定异构探针，不继续加epoch|
|输入/标签或数值合同失败|实验不可解释|修工程来源并重新完整预检；不生成科学结论|

不追加随机标签记忆训练：它即使通过也不证明真实标签含信息。本步6个模型是封顶，不追加多组tiny挑成功例。

## 6. 保存与反证

保存`TINY_IDS.json`、`INPUT_COLLISIONS.json`、六run manifest/checkpoint/history、四固定时点全动作分数、`FIT_DIAGNOSTICS.json`与独立`REVIEW.json`。报告明确pool seed42与初始化三次的不同含义。

同助手反方检查：①小样本可被记忆，阳性不可代替hold；②BCE下降可能仅学基率，必须看窗口决策；③微小连续改善标签不等于rescue，必须拆分。没有外部独立审稿。
