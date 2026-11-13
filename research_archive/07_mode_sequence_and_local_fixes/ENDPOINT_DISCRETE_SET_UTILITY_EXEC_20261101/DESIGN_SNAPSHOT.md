# 有限端点动作的集合效用学习：下一轮实验设计

日期：2026-11-01  
状态：**DESIGN_ONLY，未实现、未预检、未训练**。本文中的模型、门限和流程是下一轮预注册草案，不是已验证结果。当前交付不启动任何实验或自动任务。

## 1. 本轮要回答什么

核心问题是：**不再先回归连续端点、再量化成动作，而是直接评价原有有限动作相对“不修改”的集合收益，能否学到既救回失败窗口、又保留原成功窗口的决策？这种决策能否迁移到其他行人轨迹？**

本轮分两层回答，不能互相替代：

1. 小样本训练内：固定有限动作中的安全机会，是否能由共享轻量网络拟合出来？
2. 条件启动的迁移验证：显式历史是否比仅有原生候选几何更有用，收益是否跨轨迹保留？

这不是继续扫描 RS 的步长、保护系数、网络宽度或 epoch，也不是直接改旧 scorer。它是一个新的、有限的**动作决策表示**分支：预测动作效用，而不是预测目标坐标。

### 1.1 为什么从这里出发

已完成实验提供的证据如下，均来自原 tiny128/97 tracks，不是新泛化结果。

| 已有证据 | 数值 | 支持什么／不支持什么 |
|---|---|---|
| Native | 69 个成功窗口 | 本轮训练内参照 |
| O0 安全 GT 查表见证 | 90 成功、21 rescue、0 lost，完整组误差与运动门通过 | 有可用动作组合；不证明输入可辨识或网络可学习 |
| F-RS 三 seed | 最终全部 Native identity；各接受 6/5/9 次 | 参数变动不等于执行动作改变 |
| 真实 trial 拒绝 | 6021 次，全部仅连续原支持门失败 | 本条优化路径被连续约束卡住，不是最终 S 物理门拒绝了一切 |
| 最大归一化控制 | 约 .0497–.0533，identity 轴向最近量化边界 .125 | 尚未跨过执行动作边界 |
| 旧 Soft-RS | rescue 9/9/10；lost 8/6/6 | 可以产生收益，也会损伤已有成功；不能只优化净覆盖 |
| U2 | 12 个快照的完整提议和局部缩步均 0 rescue | 只否定这些停滞点的一次提议，不否定所有门控或其他方向 |

依据：[最终报告](../qoder_work/ENDPOINT_FEASIBLE_UPDATE_DIAGNOSIS_EXEC_20261031/RESULT_ANALYSIS_FINAL.md)、[上一轮集合目标设计](ENDPOINT_SET_UTILITY_RETENTION_DESIGN_20261030.md)。

### 1.2 归因边界

相较旧 RS，本设计同时改变了输出表示、监督目标和“最多一个槽修改”的执行结构，因此**不能把相对旧 RS 的差异纯归因于某一个因素**。旧 RS/F-RS/O0 是历史诊断参照，不是同协议因果对照。

本轮真正的配对对照只有新 G 与新 H：相同架构、动作、目标、数据、初始化、训练步骤；只改变显式历史输入是否保留。

## 2. 固定动作空间与执行合同

沿用原 END49，不新增动作、扩大范围或扫描网格：

\[
\mathcal A=\{(i/10,j/10):i,j\in\{-4,\ldots,4\},\ i^2+j^2\le16\},\quad |\mathcal A|=49.
\]

单位为米；原 action index 顺序从既有源码继承并绑定 SHA，不自行重新排序。身份动作是 `(0,0)`。对槽 k 和动作 a：

\[
P^{k,a}_t=N^k_t+\frac{t}{12}a,\quad t=1,\ldots,12.
\]

每个窗口的决策集合是：

- 一个全局 NOOP：完整 Native20；
- 20 个物理槽 × 48 个非身份动作，共 960 个单槽编辑；
- 总计 961 个决策选项，但输出始终是 **K20**，不是输出 961 条轨迹。

每个编辑保留另外 19 条原生路径逐字节不变。无连续提议 C、无最近邻量化 Q、无 STE，也不设置一条必须经过的连续安全路径。本设计只对最终选择的有限动作承担安全要求；这不修改旧实验的 C 门或回填旧结论。

### 2.1 安全掩码与选择

原 END 的七位 guard 原样继承：五项 candidate 运动量、原 junction 合同与 finite 位；各项公式、人口、零分母与 literal `native×1.01` 规则，以原锁定实现为准，禁止重释为新容差。掩码仅依赖 past/native/action，不接收 GT、group、scene、ID 或 logit。NOOP 永远合法。

对合法编辑预测标量 score，NOOP score 固定为 0。选择全局最高 score：

1. 最高值不严格大于 0，选择 NOOP。
2. 正分并列按物理槽升序、原 action index 升序选择。
3. 每个窗口最多一次编辑，不二次搜索、不按 GT 退回、不换另一个动作补救。
4. 输出后用原 guard 防御性复核；若与预先掩码不一致，输出 NOOP 并记录 `GUARD_IMPLEMENTATION_MISMATCH`，停止科学推进排查，而不是把此回退当正常成功政策。

候选级 guard 不保证全人口 pooled p95，也不保证覆盖保留。最终的组误差、完整同路径覆盖和 pooled 运动学仍必须独立评价。

## 3. 输入、模型与 G/H 对照

### 3.1 输入只包含观测与原生集合

对每个候选槽构造 48 维特征：

- 原始历史 8×2＝16 维；
- 该原生候选 12×2＝24 维；
- 该槽相对其他原生槽的 8 个最近几何距离＝8 维。

最后 8 维固定定义：

\[
D(k,j)=\max\left(\frac{\operatorname{mean}_t\|N^k_t-N^j_t\|}{.10},
\frac{\|N^k_{12}-N^j_{12}\|}{.20}\right).
\]

排除自身，19 个距离按 `(D, physical_slot)` 排序，取前 8 个。不重新聚簇、不用 GT、不把距离小等同于“可以安全删除”。这是有限集合上下文摘要，不是完整集合信息；其压缩可能仍有损。

原 40 维沿用已有 train3103 的冻结统计。新增 8 维仅用同一 train3103×20 拟合 float64 mean/std，`std<1e-6→1`；不使用 tiny、val、hold、inner 重新拟合。输入归一化后转 float32。全部坐标系和单位沿用原缓存。

| arm | 历史16维 | 原生路径24维＋集合距离8维 | 其余 |
|---|---|---|---|
| G：几何对照 | 标准化后置零 | 保留 | 与 H 相同 |
| H：历史增强 | 保留 | 保留 | 与 G 相同 |

原生路径本身已由 F16 编码过历史，所以 G 不是“完全没有历史信息”。H−G 仅检验在既有候选条件下，额外显式历史的增量价值。

### 3.2 轻量动作效用网络

共享逐槽 MLP：`48 → 64 → 32 → 48`，两层 GELU，末层线性。每槽输出 48 个非身份动作分数；NOOP 无独立可训练参数。

参数量：`3136 + 2080 + 1584 = 6800`。输出层初始化为零，故初始决策全部 NOOP。G/H 同 seed 全参数初值逐字节配对；G 被清零的历史列没有输入梯度贡献是预期行为，不伪造“全参数都有有效梯度”。

只计算矩阵乘法、忽略 bias/激活的命名口径下，单窗口 20 槽 MLP 为 133120 MAC。这不是整管线 FLOPs：原生集合距离、960 个 guard、数据移动和输出构造必须计入实际延迟。不能凭 6800 参数宣称整体轻量化成功。

原 F16/encoder/decoder/scorer 不训练、不重新实例化前向；使用兼容的冻结原生缓存。不得增加 GT、未来动作、oracle index 或样本身份嵌入。

## 4. 监督不是唯一 oracle 类别，而是完整集合的动作效用

### 4.1 同一条候选的质量

\[
J(P,Y)=\max(ADE(P,Y)/.20,FDE(P,Y)/.40),\quad good(C)=\mathbf1[\min_{P\in C}J(P,Y)\le1].
\]

禁止把不同候选的 minADE 和 minFDE 拼成一个合格候选。

设原生集合质量为 `qN=min_j J(N_j,Y)`。单槽编辑的集合质量为：

\[
q_{k,a}=\min\{\min_{j\ne k}J(N_j,Y),\ J(P^{k,a},Y)\}.
\]

这一项是关键：即使被修改候选自身变坏，只要另一个保留候选仍支持成功，就不应错误标为集合 lost；反过来，删除唯一好候选必须承担集合损失。

### 4.2 固定效用标量

本轮固定下式，不扫描系数：

\[
v_{k,a}=(\mathbf1[q_{k,a}\le1]-\mathbf1[q_N\le1])
 +\tfrac12[\operatorname{clip}(q_N,0,2)-\operatorname{clip}(q_{k,a},0,2)].
\]

- NOOP 的 v 恒为 0；范围为 [-2,2]。
- rescue 的 v>1，lost 的 v<−1；同覆盖状态的连续质量改善不能抵消一次 lost 的负号。
- 所有合法动作均有标签，多个好动作不压成唯一 oracle index；非法动作排除于损失，不被当成劣质标签。
- clip 只定义训练效用的有界辅助项，不改变科学 J、覆盖阈值或评价样本。当原／新 q 都大于 2 时，真实误差变化可能被压成 0，必须报告此类比例。
- 标签仅用于 train 监督及封存后的评价，不进入观测推理。

这仍是设计的代理效用，不等同于全部组误差、物理安全或真实未来概率。

### 4.3 固定训练损失

每个窗口将合法非身份动作按 v>0、v<0、v=0 分为三类。对每个非空类先平均 `(score−v)^2`，再对非空类等权平均，最后对 batch 中全部窗口平均。没有合法编辑的窗口损失为 0，但保留在分母和数据记录中。v 的符号按保存 float64 精确判断，不设置临时 epsilon。

目的：避免大量无效或中性编辑主导回归。代价：类别均衡重加权改变了目标分布，因此分数**不是校准的条件期望或概率**。阈值 0 是固定操作规则，不能在 val/hold 上调到看起来成功。需同时保存未加权 MSE、三类 MSE、各类计数、选中动作真实效用和 NOOP 比例。

全部损失算术在 float64 比较，模型输出 float32 后显式提升，梯度回到原参数。无 argmax 反传、无端点 MSE、无 RS/physical/protection 附加损失、无 soft winner。不得默默把旧 loss bundle 混进来。

## 5. 有限阶段与停止条件

### D0：标签／执行合同审计，不训练

继承原 source manifests 和最终封存收据，新增来源只扩展、不覆盖旧 digest。复用已审核 END train/val 动作 J 与 guard；缺失字段明确新算来源，不冒称已存在。源缺失若来自合法 NPZ 归档，先按归档清单定位，不能当作需重训。

必须完成：

1. 原 tiny128 全部 ID、97 tracks、Native69、END49/action order 和 mask 一致；O0 原查表方法仍能复现 21 rescue/0 lost，旧报告不改。
2. 新单槽集合标签用另一实现逐动作构造 K20 比较公式，覆盖唯一好槽被毁、两个好槽、重复候选、NOOP、零运动、边界相等、完全没有合法编辑等夹具。
3. GT 改变只能改变标签，不改变 features、mask、scores 或决策函数；禁止观察标签选择输入样例。
4. 保留所有合法动作的效用集合，不把 O0 唯一选择直接作训练分类标签。
5. 新效用 oracle 与旧 O0 分开命名。两者最多可救回窗口相同，不意味着选择相同、组误差／pooled motion 一样安全；不能预填“新 oracle 全门通过”。

任何来源、ID、单位或动作合同不一致：STOP，保存差异，不换 split/阈值使其通过。

### T：先做六个短训练内诊断

| 项目 | 固定内容 |
|---|---|
| 数据 | 原封存 tiny128，所有窗口／候选保留 |
| 模型 | G/H × seeds42/43/44，6 个 fresh 模型 |
| 优化 | AdamW lr=.001、weight_decay=.0001、grad_clip=5 |
| 批次／长度 | batch32；256 更新＝64 遍 tiny；固定末步为 primary |
| 配对 | 相同 seed 初始化、batch ID 顺序、训练时长；不 warmstart 旧模型 |
| 保存 | update0/64/128/192/256，全部保存点报告，不挑最佳 |

H 是事先指定的主方法，不允许 H 失败后临时改用最好 G 作为成功方法进入正式阶段。G 是诊断对照，不要求它一定失败。

进入下一阶段须 H 三个 final **同时**满足：

- 原好窗口 lost=0；
- rescue 分别不少于旧 Soft-RS 的 9/9/10，且所选动作确实非身份；这是“保留原收益并消除训练内损伤”的预算门，不是统计显著性；
- overall/stable/transition minADE 与 minFDE ≤ Native×1.01；change 两者不恶化；
- 全 group 五 pooled motion 与原 junction mean/p95 均 literal≤Native×1.01；空组标 NA，不伪造 PASS；
- 完整保存算术、source/ID/输出与 optimizer/RNG 审核通过。

三 seed 任一不满足，结束为 `TINY_ACTION_UTILITY_STOP`。不得通过延长训练、换 seed、调零阈值、换 v 系数或增加网络层现场寻找 PASS。训练内失败只否定此固定方案，不证明输入无信息。原端点回归“90% 拟合门未过”的历史结论保留，不用新目标的 MSE 替代完成。

### F：仅 T 全过且成本准入后，做六个迁移模型

| 人口 | 窗口数／change数 | 用途 |
|---|---:|---|
| train | 3103／721 | 原统计与全部效用监督 |
| val | 993／256 | fixed-final 诊断，不挑模型、阈值或系数 |
| hold | 3825／1023 | 主评价，六 final 全部封存后才开放本轮 GT 分析 |
| inner | 1856／497 | 次级迁移与成功保护检查 |

沿用原 split IDs，不重划。执行前独立列出 sequence|target track 交集、已知窗口重叠和场景组成。若与原分割合同冲突则停止；不得未经核查声称 target-disjoint 或独立测试。数据历史上已有探索暴露，本轮封存不能抹去该事实。

G/H ×3 seeds，从配对初值重新训练；batch32，固定8epoch，每epoch97更新，末批31；776更新／模型，共4656更新。T 共1536更新，若全流程触发科学更新总6192，最多12个科学模型。真实预检／benchmark 的优化次数另列，不混科学训练。

全部六个 final8、source、split、stats、动作合同与 checkpoint SHA 封存后，才开启 hold/inner 标签生成与比较。不得使用 hold 的 GT oracle 进行训练或挑动作。已有 train/val 标签复用；hold/inner 缺失的全动作效用评价成本必须计入预算。

## 6. 评价：学会动作、历史增量、实际可用分别结案

### 6.1 每个模型必须报告

- 原生与执行 K20 的同路径覆盖数、rescue、lost、retained、各自分母；不能只报告净增。
- minADE、minFDE、minJ；单 J 最近候选的早／中／晚误差，不拼分段 oracle。
- 合法编辑率、NOOP率、选中 slot/action、正／负／零真实效用、误选有害动作率、所有并列计数。
- 有限动作 regret：`max(0,max_valid v) − v_selected`。这是同一动作空间内的决策缺口，不是世界真实未来的 Bayes regret。
- 若 oracle 最大 v>0，选到任一最大效用动作均算正确；不用唯一 oracle 类别准确率处罚等价动作。另报精确并列和 near-tie 描述分布，不凭 near-tie 改主标签。
- 原 GT-free K20 complete-link `.10 meanpath/.20 endpoint` 的 distinct/effective-good，完整五 pooled motion、junction、组误差以及字节固定19槽。
- 整体、stable、transition、change 全部报告；`.3/.5` 仅固定敏感性，不能取代主 `.2/.4`。

### 6.2 正式阶段五项预注册比较

| 编号 | hold 人口／比较 | 预期方向 |
|---|---|---|
| H1 | overall regret：H−G | <0 |
| H2 | change coverage：H−G | >0 |
| H3 | change coverage：H−Native | >0 |
| H4 | change minADE：H−Native | <0 |
| H5 | change minFDE：H−Native | <0 |

固定 whole sequence|target track 配对 bootstrap，20000 draws，RNG20261101；各方法共用 track 与配对 seed 抽样索引，同 seed 对比。五项家族双侧 Bonferroni 区间分位点 `.005/.995`，并列普通95%描述区间。无分母 draw 记 invalid，有效不足19000则 `NOT_ESTIMABLE`，不补零、不补抽到显著。另报三 seed 原始效果、仅抽 track 的敏感性；3 seeds 不伪装独立样本数。

显式历史增量仅在 H1/H2 三 seed 同向且两项校正区间均排除0时成立。H≈G 只能说明本特征／容量／预算下未检出历史增量，不证明原历史没有信息；G 本身含 F16 历史处理后的 native geometry。

### 6.3 下一步推进门与原 major 门

H 须全部满足：hold change coverage 三 seed 正增、平均至少+2pp且 H3 校正下界>0；hold/inner 每 group 每 seed lost/native-good≤1%；protected overall/stable/transition minADE/FDE≤Native×1.01、change 不恶化；所有五 pooled motion 和 junction literal≤Native×1.01；inner change coverage每seed≥Native。原好分母为0则 NA，不能视为安全已证。

这只是有限推进门，不是重大突破。原 major 仍单列：change ADE/FDE 各改善≥5%、三 seed 一致、相应校正区间排除0、所有保护／覆盖／运动与相关性审核同时通过。不能用2pp门替代5%门，也不能执行成功即升级 baseline。

若 H 安全有收益但 H−G 不成立，结论为“候选几何条件动作决策可用，显式历史增量未证实”，不包装为理解运动。若 H−G 成立但最终损伤仍超限，结论为“有可迁移决策信号，但集合保护未解决”。

## 7. 核心代码接口草图（未实现、未测试）

以下仅规定职责与张量语义，不是可直接运行的生产脚本。既有 reader、guard、聚簇和统计工具需 SHA 兼容导入，不能复制后悄悄改规则。

```python
# Shapes: raw40[B,20,40], context8[B,20,8]
# valid[B,20,48], value[B,20,48], score[B,20,48]

class ActionUtilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 48))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, normalized_features):
        return self.net(normalized_features)  # no GT/ID/logit

def make_set_targets(j_native, j_action):
    # j_native[B,20]; j_action[B,20,48], each J uses SAME path.
    other = j_native[:, None, :].expand(-1, 20, -1).clone()
    idx = torch.arange(20, device=other.device)
    other[:, idx, idx] = float('inf')
    q_without = other.amin(-1)
    q_native = j_native.amin(-1)[:, None, None]
    q_edit = torch.minimum(q_without[..., None], j_action)
    delta_good = (q_edit <= 1).double() - (q_native <= 1).double()
    value = delta_good + .5 * (
        q_native.clamp(0, 2) - q_edit.clamp(0, 2))
    return value  # float64 reference labels; mask kept separately

def utility_loss(score, value, valid):
    # Invalid labels must be finite placeholders before this function.
    error = (score.double() - value).square()
    numer = torch.zeros(score.shape[0], device=score.device,
                        dtype=torch.float64)
    groups = torch.zeros_like(numer)
    for sign_mask in (value > 0, value < 0, value == 0):
        mask = valid & sign_mask
        count = mask.sum((1, 2))
        subtotal = torch.where(mask, error, 0).sum((1, 2))
        numer = numer + subtotal / count.clamp_min(1)
        groups = groups + (count > 0).double()
    return (numer / groups.clamp_min(1)).mean()

def select_one(score, valid):
    # Flatten order: physical slot ascending, original nonidentity index.
    edits = score.masked_fill(~valid, -float('inf')).flatten(1)
    no_op = torch.zeros((score.shape[0], 1), device=score.device,
                        dtype=score.dtype)
    # First-max gives NOOP precedence on exact zero ties.
    return torch.cat([no_op, edits], dim=1).argmax(1)
```

`apply_selected(native, choice, action_table)` 只接受原生路径、索引与原动作表；choice0返回原数组完整复制，其余只替换一个槽。GT 读写应与观测 inference 对象隔离。所有 score 必须 finite，否则失败，不让 NaN 通过 argmax。

建议未来代码模块：

| 模块 | 边界 |
|---|---|
| `contracts.py` | 来源／ID／单位／split／动作序／安全规则 |
| `features.py` | 原40维＋仅train拟合的8维集合几何；G/H切换 |
| `label_set_utility.py` | 分块单槽效用、valid与oracle机会，GT仅此处及评价 |
| `model.py` | 6800参数共享MLP，不含部署GT规则 |
| `decode.py` | 原安全mask、全局NOOP、最多单槽编辑 |
| `preflight.py` | 独立算术、梯度、恢复、资源与保存链 |
| `run_tiny.py` / `run_formal.py` | 先T门、后预算门，不自动重试扩展 |
| `audit_saved.py` | 独立CPU全输出审核，不调用模型前向 |

## 8. 启动前必须完成的实际预检

这不是“读源码即可 PASS”的清单，未来执行者必须产生真实收据。

1. **数值与梯度：**两arm实际GPU8×20槽，合法参数分阶段finite；原生identity、配对参数／统计、mask、tie顺序；独立NumPy标签和loss；固定非零权重、固定双eps `1e-4/1e-5` 参数方向有限差分，记录非光滑点，不搜索eps求PASS。argmax无梯度是设计事实。
2. **两轮恢复：**固定hash513train＋12val，batch32真实两epoch34updates，第19后中断；参数、外置统计、optimizer、TorchCPU/CUDA、loader、Python、NumPy RNG、sampler、batch顺序与history EXACT。没有模型buffer就如实记0，不虚称buffer恢复；这是子集测试，不是全train两轮。
3. **坏来源拒绝：**实际reader拒绝source/split/stats/actionorder/checkpoint/output损坏；完成结果只能在完整SHA一致时复用。不把analysis-skip称optimizer恢复。
4. **完整实际保存：**两个arm实际128输出，在科学启动前独立CPU重算samepath质量、原SciPy完整K20聚簇、每group pooled motion和junction、19槽不变、动作选择、效用regret。不能到结案才补上“前置已通过”。
5. **训练输入合法性：**历史清零在归一化之后；hold/inner不拟统计；逐track分割交集审计；实际改变GT夹具后预测与mask EXACT。
6. **科学源封存：**锁定后不可静默编辑或覆盖digest。必要基础设施修复须保留旧源／收据、明确影响面并重过对应门；科学协议变化必须另立版本，不借“修bug”改变目标。

## 9. 资源、时间与工件

资源沿用最新 **warning-first v3 / inventory E** 合同，不恢复过期固定6GiB门。先精确找到该资源实现与继承来源，记录估计E、进程自身private/RSS与生命周期口径；超过预警值按原合同处理，不静默提高E、不调整OS/pagefile或关闭用户程序。预计无法安全完成则停，不盲启动。

所有动作标签按window/slot分块，只保存 J/mask/value 等小张量；不物化 `N×960×20×12×2` 全套集合。审计需要完整路径时逐块重构。全部保存与恢复都计入峰值和磁盘估算。

先测实际GPU训练、20槽分数、集合距离、960动作guard、输出构造、fullquality/聚簇/CI、sourceSHA及读写，再给ETA。**T与F分别以保守完整阶段预算≤30min为启动条件**，不能只计算训练时间。超过则报告实际估计并停止该阶段，不能自动变成数小时任务、降低epoch或只审好结果。正式标签审计可能比训练更贵；目前没有可承诺的生产ETA。

后续若获执行指令，采用原唯一worker/一次startup/一次50%预计elapsed→一次完成检查流程；不重开旧任务，不恢复旧PAUSED，不高频对话轮询。当前文档阶段不创建worker或automation。

建议新执行根：`qoder_work/ENDPOINT_DISCRETE_SET_UTILITY_EXEC_20261101/`；目前不创建它、不填运行态。

每run最低工件：checkpoint、run_manifest、stats与split/actionorder SHA、optimizer与全部RNG、完整history、每窗口score/valid/choice/value/decomposition、最终K20保存输出及done SHA。checkpoint用原子提交，不覆盖失败证据。

完成必须 source/ID/model/output SHA、全部保存算术、全组质量／运动、真实资源和执行计数通过。CPU saved审核不是所有模型梯度二次回放，不是外审；预检更新、科学更新、失败前更新和重复读回分别计数。

## 10. 如何根据结果选择大分支

| 观察 | 可支持的结论 | 下一方向，不自动执行 |
|---|---|---|
| T仍不能同时救回和保留 | 当前动作效用表示／监督／有限训练不够 | 停止这一小端点分支；重新比较更高层运动状态／条件表示，不继续小系数扫描 |
| T通过，F的G/H都失败 | 训练内动作可拟合但迁移不足 | 优先研究状态条件分布和跨轨迹表示；不归咎量化卡住 |
| G/H均可迁移且相近 | 原生集合几何足以支持部分决策，显式历史增量未证实 | 保留更简单方案，避免“理解历史”的过度叙事 |
| H−G有信号，H仍lost或motion失败 | 历史含决策增量，但可靠执行不足 | 区分动作预测误差和最终集合风险，不放宽安全门 |
| H通过迁移、保护和历史增量 | 支持有限预算下可学习的互补单槽编辑 | 再申请独立数据／外部泛化与整体轻量性能验证，不直接宣布大突破 |

即使本轮失败，也不能推出“未来从当前输入不可能辨识”。有限网络、8维集合摘要、有限优化和效用重加权都可能限制结果。若成功，也只说明这些候选和数据范围内的动作效用可学，不等于理解心理意图或整模型未见泛化。

## 11. 同助手反方检查（非外部独立评审）

**DA1：是否只是把问题改成更容易的任务？** 单槽修改比旧多槽映射保守，且增加了集合距离输入，因此历史比较不具纯因果性。本轮只以新G/H作为配对归因；O0只作机会见证，不作部署胜利。

**DA2：960选1会不会挑中预测噪声？** 会，最大值选择会放大误估，NOOP阈值0也不保证校准。必须报告真实负效用选择、score分布、选中／全动作误差与rescue-lost；失败后不得在同hold上调阈值挽救。类别平衡分数不能叫未来概率。

**DA3：安全、轻量、历史理解会不会被夸大？** 候选guard不保证pooled安全；6800参数不包含guard的执行成本；H−G不是整个历史有无实验；旧数据重复使用和相关轨迹仍在。全部原门及限制必须与正结果一起交付。

## 12. 文档交付边界

本轮仅交付这份新设计和接口草图。没有实现上述模块、跑GPU夹具、读取held标签、训练模型、改变原科学源或启动自动推进。下一次实施先完成D0与实际预检；T未过则不进入F，F未获资源准入则不长时间运行。
