# 核心代码设计与实施验收

2026-10-27｜以下为实现指导片段，不是已创建、已测试或已运行的生产代码。科学定义以[D1主协议](D1_PAIRED_OBJECTIVE_DESIGN.md)为准。

## 1. 文件边界

建议未来新execution目录包含：

```text
protocol.json                  # 锁定I/S/D、全401、所有门与数据来源
source_manifest.json           # 继承旧来源digest，不覆盖
saved_score_audit.py           # D0，只读分数、标签、质量
objective.py                   # I/S原函数隔离复用；唯一新增D
train_paired.py                # 9个fresh配对终态
evaluate_saved.py             # 无GT推理；GT仅后续评价
preflight.py                   # 数学、实际梯度、恢复、保存审核
completion_review/audit.py     # 独立CPU保存数据复算
runs/{I,S,D}_seed{42,43,44}/
populations/{arm}_seed{seed}/{opt,hold,inner}/
failures/                     # 任何失败保留，不覆盖
```

旧路径不能写入；不得执行历史模块顶层目录写入逻辑。需要兼容shim时只隔离类/纯函数，并对比AST或函数内容；不把改变目标掩饰为基础设施修复。

## 2. 原网络不变

网络精确复用原324→32→32→1 GELU、11488参数、末层无bias/零初始化、原统计与STOP位处理。输入构造不接受未来GT、B/T、group或oracle action。action_id只确定既有物理槽映射和tie顺序，不新增ID embedding。

I/S的loss和choose直接依据[原editor.py](../../qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026/editor.py)核验。不能把I改成普通全动作均值BCE，不能把S改成softmax/listwise或只看抽样动作。

## 3. D损失：避免空集合与无穷值传播

```python
import torch
from torch.nn import functional as F

def decomposed_loss(scores, eligible, B):
    # scores: [batch, 401], finite, physical action order
    # eligible/B: bool; STOP eligible; B never includes STOP
    assert scores.shape == eligible.shape == B.shape
    assert eligible[:, 0].all() and not B[:, 0].any()
    assert not (B & ~eligible).any()

    edits = eligible.clone()
    edits[:, 0] = False
    positive = B & edits
    negative = edits & ~positive
    has_edit = edits.any(dim=1)
    has_positive = positive.any(dim=1)
    has_negative = negative.any(dim=1)

    # graph-connected zeros; even all-empty batches have valid zero gradients
    gate = scores[:, 0] * 0.0
    rank = scores[:, 0] * 0.0

    rows = torch.where(has_edit)[0]
    if rows.numel():
        ss = scores[rows]
        best_edit = ss.masked_fill(~edits[rows], -torch.inf).max(dim=1).values
        margin = best_edit - ss[:, 0]
        gate[rows] = torch.where(
            has_positive[rows], F.softplus(-margin), F.softplus(margin)
        )

    rows = torch.where(has_positive & has_negative)[0]
    if rows.numel():
        ss = scores[rows]
        pos = ss.masked_fill(~positive[rows], -torch.inf).max(dim=1).values
        neg = ss.masked_fill(~negative[rows], -torch.inf).max(dim=1).values
        rank[rows] = F.softplus(1.0 + neg - pos)

    per_window = 0.5 * gate + 0.5 * rank
    return per_window.mean(), {"gate": gate.detach(), "rank": rank.detach()}
```

仅在集合非空的行计算max，不采用先计算`-inf - -inf`再乘0的实现。`max(dim=1)`使用原动作顺序首个最大值，正负并列须保存；不能随机选GT有利的并列动作。

本设计不detach `best_edit`：gate梯度会流入最高编辑与STOP，共享隐藏层。因此任务分解不是参数独立，不声称消除了所有梯度干扰。

## 4. 无GT推理

```python
def choose(scores, eligible):
    assert eligible[:, 0].all()
    assert torch.isfinite(scores).all()
    # action0 first => tie chooses STOP
    return scores.masked_fill(~eligible, -torch.inf).argmax(dim=1)
```

对action>0，`r,l=divmod(action-1,20)`，只将native[r]替换为KIN[l]；action0保持native20逐位不变。不使用B、GT、组别或预测未来来决定执行；eligible沿已有GT-free物理准入。

为审计计算margin时用保存分数转float64，不改变实际训练/推理的float32分数。选择使用比较/argmax，不因margin打印四舍五入改变tie。

## 5. D0事件计算片段

```python
def diagnose(scores, eligible, B, quality, native_good):
    # Offline only: B and quality require GT, never deployment inputs.
    edit_mask = eligible.copy()
    edit_mask[:, 0] = False
    has_edit = edit_mask.any(axis=1)
    best = np.where(edit_mask, scores, -np.inf).argmax(axis=1)
    best[~has_edit] = 0
    row = np.arange(len(best))
    margin = np.full(len(best), np.nan)
    margin[has_edit] = scores[row[has_edit], best[has_edit]] - scores[has_edit, 0]
    best_is_good_edit = has_edit & B[row, best]
    best_is_rescue = best_is_good_edit & (~native_good) & (quality[row, best, 2] <= 1)
    blocked_good = best_is_good_edit & (margin <= 0)
    blocked_rescue = best_is_rescue & (margin <= 0)
    return best, margin, blocked_good, blocked_rescue
```

`quality[...,2]`必须通过manifest确认为J，不按位置猜字段。无eligible编辑时rank/margin为NA，并单独计数。所有条件事件同时保留全窗口分母。

## 6. 必须通过的预检

### 来源与数据

- 全部opt/hold/inner显式ID及track、native/KIN/标签/eligible/action映射/原统计SHA；不得仅检查长度。
- B⊆eligible，STOP不在B，T与B/STOP关系EXACT；原12/11/10物理重构、同candidate覆盖、原生保留定义。
- 三arm相同初始权重/bool/stat buffers及batch顺序；没有warmstart旧终态。
- 原I/S实际固定权重输出、标量loss及所有参数梯度与原函数EXACT，确认只D目标新增。
- 原完结I/S与新fresh控制的终态/保存输出作桥接核对；非预期差异不得作为新收益，先定位来源、顺序、运行时与数值原因。不能覆盖旧结果，不能反复重跑挑一致seed。

### 损失夹具

覆盖E空、B空、E全部为B、P/N均非空、STOP最高/编辑最高、positive/negative tie、只有1个编辑、全部scores0、有限大正负分数。

- D与独立NumPy逐行标量/score梯度复算，float64原容差1e-10。
- 对全score加相同常数，D/S目标不变、argmax不变；I不要求此性质。
- 当B非空、STOP分数最高时，D的gate对STOP存在向下梯度，系数不除以负动作数量。对固定margin复制负动作不能改变gate值；rank由最高负分而非负数目决定。
- 当B空时，D压低最佳编辑相对STOP；不能错成鼓励编辑。
- 稳定tie的解析/自动梯度规则明确检查；不在tie点要求普通有限差分可导。
- 固定seed20261027、4个预登记非零参数方向，double隔离副本FD eps1e-4/1e-5；abs1e-6、rel.001，活动最大值切换则STOP审查，不搜索eps凑PASS。该夹具不证明float32量化导数。

### 实际训练与保存

- 各arm实际8个FIT batch32全401，合法梯度分阶段有限、参数计数11488；零末层首步前层梯度0不当连接失败。
- 真实FIT_opt前513固定ID、batch32、两epoch=34updates，中断于19update；连续/恢复全部参数buffers、AdamW、CPU/CUDA/loader/Python/NumPy RNG、batch顺序cursor、history EXACT。只测试子集恢复，不冒全FIT两轮。
- 完成模型SHA-skip、错误source/统计/动作池/输出/resume合同实际拒绝；不只比较两个hash是否不同。
- 实际固定opt前128窗三arm保存401scores/选择/K20、全部质量与原SciPy聚簇、pooled motion/固定19独立重算；未通过不能宣称生产已准备好。

预检不拿hold选择配置；首次问题保存源码、trace及partial，只有实施错误且不改变科学合同时可修新版本后全预检。科学源锁后不能静默编辑或换digest。

## 7. 资源与成本预检

实际每arm4warm＋32measure训练batch，固定128窗保存与独立全聚簇/五motion/统计读回吞吐，测初末相关source SHA时间及输出规模。无保存benchmark模型用于效果选择。

先按单模型、mmap驻留、Torch、候选标签/分数、审核IO列资源inventory E，再沿当前1.3动态remaining与Windows retained-peak守卫。不能用旧6GiB固定值或向上虚报E解决超限；不改OS/pagefile/关闭用户程序。磁盘计入保存all401、checkpoint/optimizerRNG及审核临时文件，全部保留。

只在真实预检和吞吐后给ETA。仅保存score诊断阶段与训练阶段的ETA分开；机器独立worker运行不等于应用关闭后也会自动分析。

## 8. 生产完成与报告

未来真实生产必须有worker/exit双SUCCESS；九个final manifest、72epoch/9216updates历史和全部optimizer/RNG；27个arm-seed-split保存人口；source、旧输入/模型/结果和新输出初末SHA；资源、全401选择、固定19、同candidate质量/全部原聚簇/五pooled motion、全部bootstrap确定性复算。

每run manifest至少绑定：sourceSHA、split/ID/统计/候选池SHA、完整loss版本、初始化、batch顺序、optimizer、动作tie规则、推理规则、最终epoch、环境版本。绝不只有参数checkpoint。

完成后新增CPU review，不新forward或重训：重构保存选择与路径、全部group指标、gate/排名分解、两个主比较的20000个实际draw。报告审核边界，不称所有模型梯度第二次回放或外部独立审稿。

最终`RESULT_ANALYSIS.md/json`必须回答：

1. D0中非STOP排序是否含信号？STOP是否挡住真实rescue？
2. D是否同时优于S和native，而不只是比I积极？
3. 收益来自提高正确编辑、还是仅增加错误/无效编辑？
4. opt与hold/inner是否一致，所有保留/保护/运动学是否通过？
5. 原major与成本是否仍失败，哪些问题未解决？

若D0来源有缺失、某run失败、CI支持不足或门未过，显式列出，不把流程完成写成科学成功。只有未来另获具体推进请求才设计后继；当前有限系列没有自动第四目标。
