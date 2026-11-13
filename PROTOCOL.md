# kinematic-traj-forecasting 统一评价协议与实验治理规范 v1.0

> **适用范围**：kinematic-traj-forecasting 仓库自本协议生效后的全部新实验（生成、选择、蒸馏、场景先验等）。旧实验不追溯修改，只在索引中标注「协议前」。
>
> **组成**：Part D 统一评价协议（评什么、怎么判）+ Part E 工程与流程治理（怎么保证结果可信）。
>
> **状态**：v1.0 · 2026-10-20 · 护栏与非劣效界值已由作者确认；MDE 待 D-0 实测后讨论（§4.5）；track 重叠检查为冻结前阻塞项（§2.4）。冻结后的任何修改都需升版本号（v1.1…）并写入变更记录，且不得对已启动的实验生效。

## 0. 为什么需要这份协议

| 现状问题 | 后果 | 本协议对应条款 |
| --- | --- | --- |
| 口径多（outer change / overall、FIT / inner / CAL），绝对覆盖率混比 | 结论互相矛盾、难以累积 | §2、§3 |
| 成功门层层叠加（覆盖、五项运动、rescue/lost、迁移门…） | 几乎所有实验都「部分通过」，无法决策 | §3 三层指标 |
| primary 只有 253 窗口 / 68 tracks，统计未纳入 seed、未做多重比较 | 大量 CI 跨 0，但报告仍写「显著」「证伪」 | §4 |
| outer 被反复用于比较和挑选方法 | 结果变成探索性，失去独立验证意义 | §2.2 数据角色、§5 阶段门 |
| D2 预处理泄漏、审计脚本直接写 PASS | 结论不可信 | §7、§8 |
| 绝对路径、300+ 根目录、文档过期 | 不可复现、新人和模型都看不懂 | §6、§9 |
| AI 代理自动实验报告夸大措辞 | 研究方向被错误结论带偏 | §10 |

---

# Part D 统一评价协议

## 1. 两套基准，各司其职

| 基准 | 用途 | 何时必须报告 |
| --- | --- | --- |
| **内部协议（Internal）**：当前冻结缓存，28,791 窗口，四划分 + 三组别 | 方法迭代与机制研究的主战场 | 每个实验 |
| **标准协议（Standard）**：ETH/UCY 官方留一场景五折（ETH/Hotel/Univ/Zara1/Zara2），观测 8 帧、预测 12 帧、K=20 | 与文献对比、论文表格 | 仅晋级阶段（§5 S4） |

两套基准的数值**永远不放在同一列比较**。

## 2. 数据划分（内部协议，冻结）

### 2.1 划分定义

沿用 `GEMINI_EXP/core/data_loader.py` 的冻结划分，以文件 SHA-256 为准：

| 划分 | 窗口数 | 来源 |
| --- | --- | --- |
| FIT | 18,909 | `splits.npz: grouped_train_index` |
| inner | 1,856 | `splits.npz: inner_cal_index` |
| outer | 1,035 | `splits.npz: outer_eval_index` |
| original_CAL | 2,879 | 索引 25912–28790 |

组别（stable / transition / change）沿用缓存中的 `change_group`。该标签可能由未来轨迹派生，**只允许用于评价分层，严禁作为任何模型或选择器的输入、训练采样权重或早停依据**（除非实验预注册为「oracle 诊断」且结果标注 GT-assisted）。

### 2.2 数据角色（核心规则）

| 划分 | 角色 | 允许 | 禁止 |
| --- | --- | --- | --- |
| FIT | 训练 | 拟合参数、归一化统计、PCA 基底 | — |
| inner | 开发集 | 调参、选 arm、早停、阈值校准、保形校准 | — |
| outer | 确认集 | 每个预注册配置**只评一次** | 根据 outer 结果修改方法、换主方法、挑 seed、挑阈值 |
| original_CAL | 封存集 | 仅 S4 晋级时解封；**每个研究方向最多解封一次** | 任何开发用途 |

> 若某次实验违反上表（例如看了 outer 后改了配额），该实验结果自动降级为「探索性」，必须在新的预注册下重跑才能用于决策。

### 2.3 划分自检（每个实验 PREFLIGHT 必跑）

- 四划分 sample_id 互不重叠；
- **track 级不重叠**：输出 FIT∩outer、FIT∩CAL、inner∩outer 的 track 数。若非 0，需在报告中写明，并将这些共享 track 的窗口单独作为一个子组报告（Q0 已显示同 track 暴露会造成虚假信号）；
- 输入文件 SHA 与协议冻结清单一致。

> 首次全面检查见 §2.4。§2.4 完成前，本节规则照常执行，但结论需带「待重叠核查」标注。

### 2.4 阻塞项：行人（track）重叠检查（交由其他 AI 执行）

> **状态：OPEN**。完成前：(1) 协议不得正式冻结；(2) 所有 S3/S4 判定必须标注「待重叠核查」；(3) 历史上基于 outer 的学习型结果（选择器、HGB、K1-S 等）一律视为可能偏乐观。

**为什么要查**

- Q0 审计中，只有训练时见过的窗口有排序信号，同一行人但未参与训练的窗口和校准集都没有信号，提示存在「同 track 暴露」带来的虚假信号；
- 当前 track 键是 `sequence::pedestrian_id`（见 `GEMINI_EXP/core/data_loader.py`），尚未核实该键能否唯一标识一个真实行人，也未核实各划分在 track 级和帧级是否不重叠；
- 如果评价窗口与训练窗口来自同一行人、且时间上相互覆盖，评价集的未来轨迹可能已经出现在训练数据里，属于直接泄漏。

**执行约束**

- 只读：不训练、不改动任何现有文件和缓存，所有输出写入新目录 `TRACK_OVERLAP_AUDIT_YYYYMMDD/`；
- 检查脚本需独立实现，记录全部输入文件与脚本自身的 SHA-256；
- 报告按「事实 / 推断 / 未证明」分节，遵守 §10 禁用措辞。

**检查项**

| 编号 | 检查 | 具体要求 |
| --- | --- | --- |
| C1 | sample_id 不重叠 | FIT / inner / outer / original_CAL 两两交集为空 |
| C2 | track 级重叠 | 对 6 个划分对，分别输出共享 track 数、涉及窗口数；按 overall 与 change 组分别统计；重点报告 outer change（253 窗口 / 68 tracks）中有多少 track 出现在 FIT |
| C3 | 帧级时间重叠 | 对共享 track，按「观测 8 帧 + 预测 12 帧」计算每个窗口的帧区间；统计评价窗口中与任一 FIT 窗口帧区间相交的数量，单独统计「评价窗口的未来 12 帧落在 FIT 窗口内」的数量（最严重的泄漏） |
| C4 | track 键有效性 | (a) 同一 sequence 内 pedestrian_id 是否被不同行人复用（检查帧不连续、位置跳变）；(b) original_CAL 的 sequence 命名是否与其他划分一致；(c) 用取整坐标哈希查找跨划分的近重复轨迹，捕捉「同一人不同 ID」 |
| C5 | 场景级分布 | 每个划分的 scene_family 分布；明确 outer 是「已见场景」还是「未见场景」评价 |
| C6 | 上游模型训练集 | native 生成器与编码器（`GENERATION_NATIVE_DECODER_SERIES`）、KIN 生成器、K1-S / HGB 选择器、Q0 各 cohort 实际使用的训练窗口，与 outer / CAL 的 track 及帧重叠 |

**交付物**

- `OVERLAP_REPORT.json`：C1–C6 全部计数，按划分对和组别分层；
- `shared_tracks.csv`：track、所属划分、窗口数、帧区间、是否 change 组、是否帧级重叠；
- `check_overlap.py`：独立脚本，可一键重跑；
- `RESULT_ANALYSIS.md`：结论与对应的处置建议。

**处置规则（预先约定）**

| 结果 | 严重度 | 处置 |
| --- | --- | --- |
| C3 > 0（评价窗口与 FIT 帧级重叠） | 严重 | 重建 track 互斥的划分，写入协议 v1.1；所有学习型历史结果标注「可能偏乐观」；新划分完成前不做 S3 |
| C2 > 0 但 C3 = 0 | 中等 | 共享 track 的窗口作为单独子组报告；PRIMARY 额外在 track 互斥子集上做敏感性分析；评估是否重建划分 |
| C4 发现 ID 复用或近重复 | 中等 | 先修正 track 键，再重跑 C2、C3 |
| C6 发现上游模型训练集与 outer 重叠 | 严重 | 对应模型在 outer 上的结果不能作为独立验证，需在新划分上重训或明确标注 |
| C1–C6 全部为 0 | — | 关闭本项，在变更记录中登记，协议可冻结 |

**验收标准**：脚本可重跑且结果一致；每个计数都能追溯到具体 sample_id 或 track；作者审阅后把本项状态改为 CLOSED。

## 3. 指标体系：一个主指标 + 护栏 + 诊断

### 3.1 好候选定义（不变）

同一条候选同时满足 `ADE ≤ 0.20 m` 且 `FDE ≤ 0.40 m` 才算好候选。严禁拼接不同候选的 ADE 与 FDE。

### 3.2 L1 主指标（唯一，决定成败）

$$
\text{PRIMARY} = \Delta\text{Cov}_{\text{change}}^{\text{outer}} = \text{Cov}_{\text{treat}} - \text{Cov}_{\text{control}}
$$

- Cov = outer change 组中至少含一条好候选的窗口比例（K=20）；
- control = **同批重跑**的对照臂（通常是 native20 或该方向的 A 臂），不能引用旧结果；
- 每个实验**只能预注册一个主假设**，对应一个 PRIMARY。

### 3.3 L2 护栏指标（必须全部满足，否则不能 PASS）

| 编号 | 护栏 | 判定（v1.0 已确认；如需修订须升版本号，且只在实验启动前生效） |
| --- | --- | --- |
| G1 | 运动学 | outer change 组五项（speed p95、accel mean/p95、jerk mean/p95）在**每个 seed** 上都 ≤ 1.01 × native |
| G2 | 整体覆盖 | outer overall ΔCov 点估计 ≥ −1 pp，且 95% CI 下界 ≥ −3 pp |
| G3 | 同一路径误差 | Q-winner（argmin ADE+0.5·FDE）的 ADE、FDE 相对 control 恶化均 ≤ 3% |
| G4 | 丢失 | outer change 中丢失 native 好窗口的比例 ≤ 10% |
| G5 | 预算 | K=20；参数量 ≤ 305k（精度主线）；如超出须预注册说明 |

> G3 用来防止「覆盖涨了但误差明显变差」的折中被当成突破（cond_boost_11_9 就属于这一类）。

### 3.4 L3 标准指标（仅 S4 晋级时）

- ETH/UCY 五折 minADE20 / minFDE20：按文献惯例计算（minADE 与 minFDE 可来自不同候选），**在表头明确标注「文献口径」**；
- 同时报告本协议的 same-path 好候选覆盖率；
- 若方法含评分器，加报 Top-1 ADE/FDE 与 hit_best_ade。

### 3.5 L4 诊断指标（只报告，不参与判定）

rescue / lost 绝对数、effective good clusters、distinct clusters、J、分段误差（1–4 / 5–8 / 9–12 步）、stable 和 transition 组、inner 与 CAL 上的同类指标、修正量 mean/p95/max。

## 4. 统计方法

### 4.1 推断单位与重抽样

- **推断单位是 track，不是窗口**。同一 track 的窗口高度相关，按窗口算 CI 会严重低估不确定性；
- **两级配对 bootstrap**：每次 draw 先有放回抽 track（取其全部 change 窗口），再有放回抽 seed，treat 与 control 使用**同一组 track 和 seed**；20,000 draws；报告 95% 百分位 CI；
- 必须报告每个 seed 的单独结果，禁止用最好 seed 代表整体。

### 4.2 seed 数量

| 阶段 | seed |
| --- | --- |
| S2 筛选 | 42 |
| S3 确认 | 42、43、44 |
| S4 晋级 | 42–46（5 个） |

### 4.3 多重比较

- 一个实验内有多个 treatment 臂时，PRIMARY 使用 Holm 校正；
- 护栏属于非劣效检验，单侧、不校正，但界值必须预注册；
- 组别、划分、指标的二次分析一律归入 L4，不得拿来宣称成功。

### 4.4 判定语言（只允许这四种）

| 判定 | 条件 |
| --- | --- |
| **PASS** | PRIMARY 校正后 CI 下界 > 0，且 G1–G5 全部满足 |
| **NON-INFERIOR** | PRIMARY CI 下界 > −2 pp，且护栏全部满足（可用于「同精度更轻量」类目标） |
| **INCONCLUSIVE** | CI 跨 0 且跨 −2 pp：**只能说「未证明有效」，不能说「已证伪」** |
| **FAIL** | CI 上界 < 0，或任一护栏失败 |

### 4.5 功效校准（D-0，协议冻结前必须完成）

**状态：待实测。**MDE 的具体取值及其对 S4 要求的影响，等 D-0 结果出来后再与作者讨论确定；在此之前 4 pp 只是暂定参考线。

现有 primary 仅 68 tracks，粗略估算最小可检测效应（MDE）约 4–6 pp，需实测确认：

1. 用已保存的 native20 与某一已完成 A/B 实验的三 seed outer 预测，按 §4.1 估计 ΔCov 的标准误；
2. 计算 80% 功效下的 MDE；
3. 若 MDE > 4 pp，S4 必须加评 original_CAL change 组（并与 outer 合并报告一个汇总估计），或扩充 seed 至 5 个；
4. 在实验设计中写明「本实验预期效应是否大于 MDE」。**预期效应小于 MDE 的实验不应启动**，或改为纯探索。

## 5. 实验阶段门

```text
S0 预注册 → S1 冒烟+因果测试 → S2 筛选(inner, seed42) → S3 确认(outer, 3 seeds) → S4 晋级(CAL+5 seeds+标准五折)
```

| 阶段 | 用哪个划分 | 输入 | 通过条件 | 产物 |
| --- | --- | --- | --- | --- |
| S0 预注册 | — | 假设、对照、唯一 PRIMARY、护栏界值、预期效应、预算 | 人工确认后锁定 `protocol.json` SHA | `DESIGN.md`、`protocol.json` |
| S1 冒烟 | FIT 小样本 | 代码 | §7 全部测试通过；初始化时输出与 control 一致（零初始化类方法） | `PREFLIGHT.json` |
| S2 筛选 | **inner** | seed42 | inner change ΔCov > 0 且 G1 不明显失败 | `screen.json` |
| S3 确认 | **outer（一次）** | seeds 42–44 | §4.4 PASS 或 NON-INFERIOR | `RESULT.json`、`AUDIT.json`、`RESULT_ANALYSIS.md` |
| S4 晋级 | original_CAL + 标准五折 | seeds 42–46 | CAL 上方向一致且护栏满足 | 更新 baseline 记录 |

规则：

- S2 可以多臂并行；进入 S3 的臂数 ≤ 2；
- S3 失败后**不得**在同一预注册内调参重跑 outer；如需修改，回到 S0 重新登记；
- 只有 S4 通过的方法才能成为新 baseline，并写入 `BASELINE.md`。

## 6. 报告模板（`RESULT_ANALYSIS.md` 固定结构）

```markdown
# <实验ID> 结果分析
## 1. 预注册摘要（引用 protocol.json SHA）
## 2. 判定：PASS / NON-INFERIOR / INCONCLUSIVE / FAIL
## 3. 主指标表（每 seed + 两级 bootstrap CI）
## 4. 护栏表（G1–G5 逐项）
## 5. 诊断指标（L4，不参与判定）
## 6. 事实（只写数字能直接支持的）
## 7. 推断（单独标注，写清依据）
## 8. 未证明 / 不能推出
## 9. 偏离协议之处（没有就写“无”）
## 10. 下一步（只能是预注册的分支之一）
```

---

# Part E 工程与流程治理

## 7. 因果性与泄漏测试（pytest，S1 必跑）

放在 `tests/`，任何新实验的数据管线和模型都必须通过：

| 编号 | 测试 | 做法 | 针对的历史问题 |
| --- | --- | --- | --- |
| T1 | 隐藏点不变性 | 随机扰动被遮挡或未来的点，断言可见输入张量逐元素不变 | D2 用隐藏点决定旋转方向 |
| T2 | 归一化只用可见点 | 对 heading / 平移 / 缩放函数，断言只读取 `t ≤ T_visible` 的索引 | 同上 |
| T3 | 真值隔离 | 推理时把 GT 和 change_group 替换为 NaN 或随机值，断言输出不变 | 选择器、门控误用 GT |
| T4 | 统计量只来自 FIT | 断言均值、方差、PCA、HGB 训练样本 ID ⊆ FIT | 评价集信息泄漏 |
| T5 | 划分不重叠 | sample_id 与 track 级重叠检查（§2.3） | Q0 训练暴露过拟合 |
| T6 | 基线复现 | native20 指标与冻结缓存逐位一致 | 对照漂移 |
| T7 | 确定性 | 同 seed 跑两次，指标差 ≤ 1e-6（或记录非确定性来源） | 不可复现 |
| T8 | 初始化一致性 | 零初始化残差 / 门控类方法，训练前输出与 control 完全一致 | 「改了结构但其实没生效」 |

## 8. 真实审计（替代「写死 PASS」）

- 审计脚本必须是**独立实现**：不 import 训练代码的评价函数，直接从保存的预测（`.npz`）与冻结 GT 重算 §3 全部指标；
- 与 `RESULT.json` 逐字段比较，记录 `max_abs_diff`；**只有 diff = 0（或 ≤ 预注册容差）时程序自动写 PASS**，状态字段禁止硬编码；
- 审计输出必须包含：被审计文件的 SHA-256、审计脚本自身的 SHA-256、比较字段数、差异字段列表；
- 覆盖范围：outer 必审；S4 时 CAL 与标准五折必审。

```python
# audit/verify.py（骨架）
res = json.load(open(result_path))
recomputed = independent_metrics(np.load(pred_path), gt, groups)  # 独立实现
diffs = compare(res, recomputed)          # 逐字段
status = 'PASS' if max(diffs.values(), default=0) <= TOL else 'FAIL'
write_json('AUDIT.json', dict(status=status, max_abs_diff=max(diffs.values()),
           n_fields=len(diffs), mismatches=[k for k,v in diffs.items() if v > TOL],
           inputs_sha=sha_all([result_path, pred_path]), script_sha=sha(__file__)))
```

## 9. 仓库结构与可复现性

### 9.1 目标结构

```text
kinematic-traj-forecasting/
├── README.md            # 只写项目简介、当前 baseline、阅读路线
├── BASELINE.md          # 当前 baseline 定义与 S4 证据
├── PROTOCOL.md          # 本协议（版本化）
├── config/
│   ├── paths.example.yaml   # 数据、外部 ASTRA 工程、输出路径
│   └── protocol_v1.json     # 冻结的划分 SHA、阈值、护栏
├── src/f16/             # 模型、数据、评价的唯一实现
├── tests/               # §7 测试
├── audit/               # §8 独立审计
├── experiments/
│   └── 2026-10-xx_<方向>_<ID>/   # DESIGN / protocol / PREFLIGHT / RESULT / AUDIT / RESULT_ANALYSIS
├── reviews/             # 外部模型或人工复核（GEMINI / GROK / Claude 等）
├── docs/                # 方法详解、调研
└── archive/
    ├── INDEX.csv        # 旧实验索引：目录名、日期、方向、结论、是否协议前
    └── legacy/          # 现有 300+ 目录整体移入，不改内容
```

### 9.2 路径与环境

- 所有脚本通过环境变量 `F16_ROOT` 或 `config/paths.yaml` 取路径，**禁止出现 C:/Users/... 这类 Windows 绝对路径**；
- 在 CI 或 pre-commit 中用 `rg -n "C:\\\\Users|/home/" src experiments` 检查，命中即失败；
- 每个实验保存：git commit SHA、`protocol.json` SHA、数据清单 SHA、`pip freeze`、全部 RNG 状态（Python / NumPy / Torch / CUDA）、含 optimizer 的 checkpoint（可不入库，但记录路径与 SHA）。

### 9.3 文档维护

- `docs/当前模型方法详解.md` 更新为 F16 + K20 当前底座，旧 Q20/E24 内容移入「历史版本」一节；
- 每次 baseline 变更时同步更新 `README.md` 与 `BASELINE.md`。

### 9.4 公开仓库安全

- 仓库为公开状态：提交前对新增文件运行密钥扫描；`email_notifier.py` 等脚本的账号与密钥只从环境变量读取；
- `.gitignore` 继续排除数据、权重与日志。

## 10. AI 代理使用规范

仓库中大量实验由 Codex / Claude / Gemini / Grok 等代理自动执行与撰写。规则如下：

1. 代理可以执行 S1–S3，但**不能修改已锁定的 `protocol.json`**，不能自行进入 S4；
2. 代理报告必须使用 §6 模板，「事实 / 推断 / 未证明」分节；
3. **禁用措辞**：「严格证伪」「唯一根因」「彻底排除」「确认性」（非预注册时）、「高度显著」「突破」（除非 §4.4 判定为 PASS 且通过 S4）；
4. 判定只能是 §4.4 四种之一；p 值必须注明是否校正、是否纳入 seed；
5. 每轮代理任务开始前写明预算（GPU 小时、实验数），超出即停止并汇报；
6. 晋级（S4）和 baseline 变更必须人工签字确认。

## 11. 研究节奏：诊断配额

近期大量精力投入在审计与诊断（W0、Q0、K0-C 等）。建议：

- 每个研究周期（约一周）中，**诊断 / 审计类实验 ≤ 50%**，其余必须是直接推进模型的 S2–S3 实验；
- 诊断实验也需预注册，并写明「结果将如何改变下一步决策」，写不出来的不做。

---

## 12. 落地计划

| 步骤 | 内容 | 预计耗时 |
| --- | --- | --- |
| 1 | ✅ 作者确认 G2–G4 界值与 NON-INFERIOR 界值（2026-10-20） | 已完成 |
| 1b | 🚧 track 重叠检查（§2.4，交由其他 AI 执行，作者审阅） | 0.5–1 天 |
| 2 | D-0 功效校准（§4.5），结果出来后与作者讨论 MDE 及是否必须加评 CAL | 0.5 天 |
| 3 | 建立 `config/`、`src/f16/eval.py`（从 `GEMINI_EXP/core/evaluation.py` 提取，加 track 级 bootstrap） | 1 天 |
| 4 | 编写 `tests/` T1–T8 与 `audit/verify.py` | 1–2 天 |
| 5 | 用 native20 与一个已完成实验（如 EXP03）回放：验证新评价与审计能复现旧数字 | 0.5 天 |
| 6 | 旧目录移入 `archive/legacy/`，生成 `INDEX.csv`；更新 README / 方法详解 | 1 天 |
| 7 | 冻结协议 v1.0，之后的方向 A / B 实验全部按本协议执行 | — |

## 附录 A：protocol.json 示例

```json
{
  "protocol_version": "1.0",
  "experiment_id": "2026-10-xx_B_joint_hybrid_K20",
  "hypothesis": "联合 native/KIN 槽位的单阶段 K20 生成器提升 outer change 好候选覆盖",
  "control": "native20 (same batch)",
  "treatment": ["joint_hybrid_11_9"],
  "primary": "delta_cov_outer_change",
  "expected_effect_pp": 4.0,
  "mde_pp": null,
  "guardrails": {"G1_motion_factor": 1.01, "G2_overall_min_pp": -1.0, "G2_overall_ci_low_pp": -3.0,
                 "G3_same_path_rel": 0.03, "G4_lost_rate_max": 0.10, "G5_K": 20, "G5_params_max": 305000},
  "noninferior_margin_pp": -2.0,
  "seeds": {"S2": [42], "S3": [42, 43, 44], "S4": [42, 43, 44, 45, 46]},
  "bootstrap": {"unit": "track", "two_level_seed": true, "draws": 20000, "multiplicity": "holm"},
  "data_manifest_sha256": "<填写>",
  "budget": {"gpu_hours": 0, "max_arms_S3": 2},
  "locked_at": "<时间戳>",
  "locked_by": "lin mocheng"
}
```

## 附录 B：术语

- **好候选**：同一条轨迹 ADE ≤ 0.2 m 且 FDE ≤ 0.4 m；
- **Cov（覆盖）**：含至少一条好候选的窗口比例；
- **rescue / lost**：相对 native 新增 / 丢失的好窗口；
- **Q-winner**：argmin(ADE + 0.5·FDE) 的候选；
- **同批对照**：与 treatment 同代码版本、同数据、同 seed 一起重跑的 control。

## 变更记录

| 版本 | 日期 | 变更 |
| --- | --- | --- |
| v1.0 草案 | 2026-10-20 | 初稿 |
| v1.0 | 2026-10-20 | 确认 G1–G5 与非劣效界值；MDE 标记为待实测后讨论；新增 §2.4 track 重叠检查阻塞项 |
