# 实施与验收合同：固定K20的一次性互补编辑

日期：2026-10-25。状态：**仅设计，未实现、未预检、未启动**。

与[主设计](01_EXPERIMENT_DESIGN.md)共同生效。下述代码是实现蓝图，不是已经验证可运行的软件。任何当前文件存在、维度合理、历史PASS，都不能替代本轮真实ID/source/数值/资源审核。

## 1. 模块边界与输入合同

未来实现只写新的执行子目录，建议模块如下；本次没有创建这些科学代码：

```text
data_contract.py       # 显式ID映射、split与输入SHA，不靠行数猜身份
candidate_metrics.py   # 同路径质量、原运动学与完整聚簇
single_edit_space.py   # STOP + 400动作，唯一GT-free物理准入
label_acceptance.py    # 训练/事后多正例标签，禁止被部署调用
feature_builder.py    # 固定324通道、opt-only统计
editor.py              # 相同MLP、I/S两个损失
run_e0.py              # 有限枚举与原机会门
train_paired.py        # 六次fresh拟合，固定epoch8
evaluate_fixed.py      # 固定终态；不做阈值/epoch选择
audit_saved.py         # 独立CPU保存算术、来源与身份复核
benchmark_pipeline.py # 缓存与端到端口径分离
```

每seed、每split加载：`sample_id[N]`、`track_key[N]`、`past[N,8,2]`、`native[N,20,12,2]`、`kin[N,20,12,2]`、冻结观测context、`gt[N,12,2]`和原group标签。IDs/groups/GT在控制或监督侧，不传编辑器。

- context必须与native对应同一原观测、编码来源和模型；使用原有96维recent与96维mean8tokens缓存。若该来源没有可验证缓存，先报告具体缺项，不能拿别的模型tokens拼接，也不能悄悄重新前向填补。
- 输入单位为米，dt=0.4秒；每窗口以最后观测位置为原点。保持原坐标朝向，不新增旋转或速度归一化。
- 三seed同split IDs/GT/分组逐值相同；native是否逐seed不同按真实来源记录，不能将seed42原生基准套到所有seed。
- 明确KIN FIT缓存的原始行空间，再按旧索引与身份连接到18,909；检查无重复、无缺失、无跨split泄漏。
- 对FIT_hold做整track排除仅约束编辑器；不得伪称冻结生成器从未见hold。
- 若所需数组已被归档，只恢复精确需要的容器并核原SHA；不全量恢复、不重训重造、不改旧manifest。恢复需要的空间另计。

新`INPUT_MANIFEST.json`记录原始与当前路径、SHA、shape、dtype、坐标变换、ID映射来源和有无历史运行锁源。现在建立SHA只能证明本轮输入固定，不能反向证明旧模型当时的训练协议。

## 2. 动作与运动学：只有一套部署规则

动作编号0是STOP；其余`1+20*r+l`，r/l各0…19。相同固定顺序用于分数并列、标签并列和保存。

```python
def materialize(native20, kin20, action_id):
    out = native20.copy()
    if action_id != 0:
        r, l = divmod(action_id - 1, 20)
        out[r] = kin20[l]
    return out

def eligible_actions(native20, kin20):
    # 接口严格只有轨迹；不接收GT、group、scene、ID、logits。
    base = five_motion(native20)
    valid = [True]  # STOP不因重新计算末位差被屏蔽
    for a in range(1, 401):
        proposed = materialize(native20, kin20, a)
        valid.append(all(five_motion(proposed) <= 1.01 * base))
    return valid
```

`five_motion`采用项目已锁定定义：速度为预测相邻点11个差分，加速度10个、jerk9个；分别除dt、dt²、dt³，L2后池化全部K×时间。speed p95、acc mean/p95、jerk mean/p95；p95为NumPy linear quantile。先转float64再差分，不添加观测末点创造第12个速度。实施前核对原科学源：若原定义与此不一致，停止并显式协调协议，不默默换口径。

优化实现可用可验证的次序统计与增删数组，但必须与独立直接重构版本逐项比较。科学比较仍literal，不把数值审核容差加进1.01门。每个拒绝保留五项bool及原值，不能只留一个总flag。

不能持久化全部`N×400×20×12×2`路径。保存两原池、动作索引、mask和数值，分块重构；oracle和最终选择的K20才保存完整路径。

## 3. 固定324维动作特征

所有轨迹均为上述物理坐标。每行代表一个窗口的一个动作，两臂使用完全相同数值。

|块|内容|维数|
|---|---|---:|
|历史|8个位置、7个速度、6个加速度，分别flatten|16+14+12=42|
|冻结context|recent96与mean8tokens96，原缓存定义不变|192|
|原生集合|逐时刻均值轨迹24；20个endpoint的总体协方差xx/xy/yy，ddof0；原生五运动量|24+3+5=32|
|替换对|被删native路径、换入KIN路径|24+24=48|
|差异|换入−被删endpoint；完整替换集合−原生五运动量|2+5=7|
|几何关系|换入与被删的D；换入与剩余19条原生的min D|2|
|类型|is_stop|1|
|合计||324|

`D(p,q)=max(mean_t ||p_t-q_t||/.10, ||p_12-q_12||/.20)`。它只是特征，不是新的novelty准入门。

STOP保留窗口公共266维；替换相关57维原始值为0；is_stop=1。替换动作is_stop=0。不加slot-ID embedding、GT距离、repairability、标签支持数、原logit、scene或track。

除is_stop外，统计仅来自FIT_opt；每窗口权重相同，窗口内部有效动作含STOP等权。使用float64加权稳定均值/方差（ddof0），std<1e-6置1，再转换模型float32。禁止用hold/inner或全25912拟合。标准化后STOP替换块不一定仍为0，这是预期，不再特殊重置；保存原始与标准化夹具。所有通道始终有限，方差负值超过浮点舍入证书即失败，不靠任意clip掩盖错误。

此特征不声称最优；E0后不能按结果删加通道。若context来源不可核验，不以“去掉context”作为同一次实验继续。

## 4. 标签：完整动作集合，不是唯一最优槽位

```python
def label_actions(native_metrics, action_metrics, eligible):
    # A=minADE, F=minFDE; J=min_k max(ADE_k/.2, FDE_k/.4)
    nonworse = (action_metrics <= native_metrics).all(axis=-1)
    strict = (action_metrics < native_metrics).any(axis=-1)
    preserve_good = ((native_metrics[2] > 1) |
                     (action_metrics[:, 2] <= 1))
    B = eligible & nonworse & strict & preserve_good
    B[0] = False
    target = B.copy()
    if not B.any():
        target[0] = True
    return B, target
```

按真实float64坐标计算ADE/FDE；不要先把逐slot误差round后比较。保存原始margin，另描述`0<改善<=1e-10`的数量，但这个描述桶不改变标签或门。

E0只对FIT_opt建立标签与机会门。只有E1完成六模型锁定终态后，才计算hold/inner完整标签作事后解释；它们不进入batch、统计、调参或模型选择。

动作错误拆分：

- 非STOP且在B：有效编辑。
- 非STOP不在B且A/F/J任一变差或丢native good：质量有害。
- 非STOP不在B且上述均未变：无效编辑。
- STOP且B非空：遗漏机会，不称伤害native。
- STOP且B空：正确保留。

任何label不自带部署可知性；“存在多个好动作”不是“观测必然可辨识好动作”。

## 5. 核心模型与损失接口

```python
class ActionEditor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(324, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
            nn.Linear(32, 1, bias=False))
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, observable_features):
        return self.net(observable_features).squeeze(-1)

def window_loss(scores, eligible, target, arm):
    pos = eligible & target
    neg = eligible & ~target
    if not neg.any():
        return scores.sum() * 0  # 只STOP可用，仍计入batch窗口数
    assert pos.any()
    if arm == "I":
        return 0.5 * F.softplus(-scores[pos]).mean() + \
               0.5 * F.softplus(scores[neg]).mean()
    # stable_first_max返回固定动作顺序中首个最大值及其梯度，
    # 不用会把梯度均分给并列项的另一种amax替代。
    return F.softplus(1 + stable_first_max(scores[neg]) -
                     stable_first_max(scores[pos]))

def choose(scores, eligible):
    masked = scores.masked_fill(~eligible, -torch.inf)
    return masked.argmax(dim=-1)  # 动作0优先，其次r/l
```

参数数为`325*32 + 33*32 + 32 = 11,488`。零末层使初始所有分数相同，推理STOP，输出native20 EXACT；首步前层梯度为0是该初始化的正常性质，需分阶段核验之后梯度连接，不虚构首步全参数非零。两臂均同初始化。

一个训练batch为32窗口，每窗口401动作；loss先逐窗口计算，再对全部32窗口均值。不得把所有窗口正负动作混在一起统一均值，或只对B非空窗口训练。feature分块可以concat成完整401分数，不能把每动作块的max/损失各算一次当成全窗口集合损失。

S的hard max活动面变化必须记录。预检double参数有限差分在固定非零夹具上用eps1e-4、1e-5；记录pos/neg winner切换，若切换不能当光滑梯度PASS，也不能看结果后搜索更小eps。可以另给解析/单侧证据，但该预注册数值门状态保持未通过，停止正式启动并说明。

## 6. 启动前必须真实完成的检查

1. 当前输入源、split/ID、坐标、context、native/KIN来源、统计与完整SHA初末核验。
2. 夹具覆盖：仅STOP有效、多个正例、B空、重复候选、native唯一好候选被删、ADE/FDE不同slot、native零运动量、五物理项逐项拒绝、分数并列STOP优先、同一GT改变不影响features/mask/forward。
3. 两臂真实FIT八个batch32的有限输出与分阶段梯度连接；paired参数/统计全同、初始nativeidentity、原池不变。不加载旧生成器优化器。
4. I/S损失与独立NumPy标量、double参数梯度夹具一致；独立排序版本明确首个并列项。数值容差须在执行protocol锁源前列出，不能以训练结果倒推。建议标量1e-10绝对、FD `abs_err<=1e-6+1e-3*abs(reference)`，这些是审核容差，不是科学效果epsilon。
5. 各臂真实FIT_opt前513窗的2epoch连续/中断续训，比较全部参数、bool/统计buffers、AdamW状态、Torch CPU/CUDA、loader、NumPy、Python RNG、history和诊断 EXACT。不是完整4096两轮或旧模型重训。
6. completed模型/analysis真实SHA-skip；篡改source、split、stat、checkpoint或输出时实际读取分支拒绝。不能只证明两个SHA字符串不相等。
7. 实际保存128窗的两臂输出，经独立重构动作/固定19条、原SciPy全K20聚簇/质量/五motion与来源SHA复核。

real-ID夹具不得按GT挑成功样本替换预注册训练子集。边界/多正例可用独立合成夹具；真实支持不足需记录，不宣称合成通过代表自然收益。失败保留source/log/fixture，修基础设施后全当前预检重做；科学规则变化则另立设计版本，不覆盖锁源。

## 7. 保存与独立验收

每run最低保存：`checkpoint.pt`、`run_manifest.json`、`stats.npz`、`split_ids.json`、`history.json`、`assignment_stats.json`、`metrics.json`、显式ID预测与动作文件、`done.json`。

checkpoint绑定source/input/split/stat/protocol/init SHA，epoch/step、模型全部参数/buffers、optimizer与ALL RNG；manifest重复这些可读字段，并列出I/S唯一差异与缓存/生成器身份。文件名、seed或checkpoint单独参数不构成协议证据。

E0每窗口保存400个有效mask、多正例mask、A/F/J及五运动量、原始margin、oracle动作与输出。按action块有界写入，不保存全部动作的完整轨迹。独立CPU遍历**全部枚举动作**重建标签和机会门，不用128夹具冒全审核；质量min可有独立top2加删实现，遇并列必须正确。oracle完整K20另做全聚簇和pooled motion。

E1/E2保存opt/hold/inner全部窗口最终401分数或可无歧义重建的分数存档、eligible/B/target、chosenaction及K20；保存每seed每arm，不只保存均值或最好seed。

completion独立CPU审核：全部最终路径、固定19条、STOP identity、动作argmax、质量与rescue/lost分母、全K20原complete-link聚簇、effectivegood、五pooled motion、标签与assignment、48epoch记录、六完整manifest/checkpointSHA、全部输入输出源SHA。bootstrap固定draws可重构，原样保存20000 pairedtrack/seed与固定seed敏感性结果。

独立saved算术不是模型/梯度第二次重放，也不是外部同行审核。不能给I/S算不同CI draws，不能把两个单独CI重叠当paired比较。

## 8. 资源、耗时与一次性工作流

本次设计不承诺未经测量的分钟数。实施前分别测：E0真实128窗全部400动作及双审核；E1各臂4warm+32measure batch32；最终全质量聚簇、CI、源SHA与IO。拆分预测时间，并实测Windows retained peak及磁盘临时空间。

按当前项目规则列资源inventory E，以1.3倍动态remaining/保留峰作为资源检查，不恢复已撤销的固定6GiB或30分钟/六小时旧硬限。用户短预算偏好仍有效：若测得成本明显超出短测试规模，先交付具体阶段成本与可停止点，不擅减样本/epoch改变科学合同，也不无界延长。

缓存推理速度与真正端到端成本分开；端到端需包括native和KIN生成、400动作物理与特征、传输及编辑器，50warm/500measure、CPU1thread与GPU各自说明batch与同步口径。计唯一参数，不把共享encoder重复相加，也不能只报11,488。

实际执行获准、全部对应门PASS后，才单一按需零trigger OS worker；启动确认一次，预计elapsed50%检查一次，按实际剩余阶段修正完成检查时间。不是每日轮询，不重复历史PID/旧任务，不恢复旧PAUSED任务。E0未过不启动E1，不能为了跑满系列跳门。

## 9. 最终报告必须回答

1. 一次替换本身是否有足够、可核验的覆盖机会？与旧多步oracle有什么区别？
2. I/S是在FIT_opt就学不会，还是在hold/inner迁移时失败？负例STOP与有害编辑是否混淆？
3. S是否在唯一预注册主比较胜I，且没有通过退化I制造胜利？
4. native成功保留、所有组五运动学和真实成本是否分别通过？
5. 哪个结论是观察，哪个是机制推断，哪个仍未验证？

交付`RESULT_ANALYSIS.md/json`＋完整验收清单，状态只可据实为E0_STOP、E1_FAILED、ANALYZED_NO_PRIMARY_EFFECT、LOCAL_DECISION_SIGNAL_NOT_DEPLOYABLE或ALL_REGISTERED_GATES_PASS。最后一个仍需保留反复开发inner、窗口相关、三个seed、冻结模型既有暴露的探索限制，不自动升级baseline或宣称理解行人意图。
