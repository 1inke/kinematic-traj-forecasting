# 端点／时间可观测性系列：代码使用与交付边界

更新：2026-10-29。对应原始设计：`01_SERIES_DESIGN.md`、`02_CORE_CONTRACTS.md`、`03_EVALUATION_AND_DECISIONS.md`。

**当前状态：CORE_IMPLEMENTED_SYNTHETIC_TESTED_REAL_PREFLIGHT_PENDING。**

本次补充18个Python模块和22项合成CPU测试；没有计算真实人口的新教师标签，没有正式训练，没有启动OS worker或自动跟进，没有恢复任何旧暂停任务。原设计文件的 `DESIGN_ONLY` 是2026-10-28交付快照，不是本次代码实现状态。

这是一套可调用的核心模块和显式阶段命令，**还不是已经通过真实预检的一键生产流水线**。第8节逐项列出尚未实现／尚未验证的生产环节，不能手写一个PASS字段绕过它们。

## 1. 模块与职责

| 文件（均在 `code/`） | 已实现内容 |
|---|---|
| `action_geometry.py` | END49、TIME441、几何合法性、精确identity、固定优先级、连续参数最近动作解码 |
| `motion_guard.py` | GT-free逐候选五运动量与历史衔接检查、逐项失败位、整槽Native回退、独立人口pooled计算 |
| `io_contract.py` | 非pickle NPZ读写、SHA、原子语义上的独占新文件创建、源／输出封存和终态复用检查 |
| `input_audit.py` | 继承原opt划分，核3103/993及721/256、sample/track不交叉、生成train-only统计 |
| `build_teacher.py` | 先封存无GT动作掩码，再单独读GT生成同槽J教师、多解／近并列／机会计数 |
| `features.py` | 共享40维输入、训练统计；G在标准化后遮掉前16维历史，H保留历史 |
| `model.py` | 4770参数的40→64→32→2 GELU/tanh共享MLP，逐窗／槽／坐标MSE |
| `train.py` | 固定8epoch训练内核、确定性采样、参数／优化器／所有相关RNG／批次顺序的精确恢复 |
| `inference.py` | 不接收GT的参数解码、raw和guarded保存；P3固定先TIME再加END偏移及联合再检查 |
| `evaluate.py` | 同槽good/J、rescue/lost、原SciPy complete-link、有效good、五pooled运动量与junction |
| `bootstrap.py` | 20,000整track配对seed抽样，六比较家族及固定seed敏感性；空group抽样不补0 |
| `audit_saved.py` | 第二套标量几何／物理算术；核最近动作、原始与执行路径、每项拒绝原因 |
| `preflight_checks.py` | 可调用的独立NumPy全梯度、固定两eps差分、真实513+12子集恢复、tiny拟合检查函数 |
| `run_stage.py` | prepare/masks/labels/fit/predict/baseline/evaluate/audit显式入口与前置门控 |
| `compare_models.py` | 按42/43/44配对读G/H保存结果，计算MSE、regret、覆盖的分组CI |
| `combine_saved.py` | 两分支正式准入后、同seed H输出的保存结果联合，不训练 |
| `freeze_sources.py` | 核旧SHA、拒绝替换旧digest，只扩展本次源与显式依赖 |
| `test_core.py` | 22项无真实数据的合成CPU测试 |

文件写入均拒绝覆盖。独占创建不等于整个多文件阶段事务原子提交；中断留下的部分文件保留，由终态封存阻止误复用。

## 2. 本次验证到什么程度

已实际运行：

- 22项合成测试全部通过；最终一次 `Ran 22 tests in 5.238s / OK`，进程退出0。该时间只属于单元测试，不是实验ETA。
- G/H合成8窗的独立NumPy双精度MSE及全部参数梯度核对、固定eps `1e-4/1e-5` 差分通过。不是实际GPU检查，也不是float32量化梯度证明。
- 合成7窗、batch3、2epoch的中途序列化／反序列化恢复：参数、optimizer、CPU及采样器RNG、Python/NumPy RNG、history和批顺序全部一致。不是实际513窗恢复，更不是全3103窗两轮。
- END49／TIME441动作、零段长／静止identity、同槽ADE/FDE、原生零阈值、单路径早中晚、聚簇重复、逐项guard、最近动作和损坏输出拒绝。
- 18个Python模块AST语法检查通过；四个命令行入口 `--help` 退出0；`run_stage.py` 不带 `--run` 时，对不存在的输入也只输出 `PLAN_ONLY`，未创建输出。

本轮使用现有 `.venv-governance/Scripts/python.exe`，NumPy 1.25.2、SciPy 1.12.0、Torch 2.11.0+cu128。模型及梯度测试在CPU执行；没有安装新依赖。系统默认Python缺SciPy，因此不要直接用未核对的 `python` 代替这个环境。

核验记录见 [05_IMPLEMENTATION_VERIFICATION.md](05_IMPLEMENTATION_VERIFICATION.md)。

## 3. 输入、输出与GT隔离

已只读核对的现有入口：

```text
qoder_work/K20_MODE_COVERAGE_AUDIT_20261027/inputs/X_opt.npz
qoder_work/K20_MODE_COVERAGE_AUDIT_20261027/inputs/GT_opt.npz
qoder_work/K20_COLLABORATIVE_REFINEMENT_RESOURCE_ADAPTER_20261027/prepared/split.npz
qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py
```

`X`最低字段：`sample_id[N]`、`past[N,8,2]`、`native[N,20,12,2]`、`track_key[N]`、`group[N]`。past末点必须精确0；group必须使用原0/1/2。`GT`单独文件，仅含对应 `sample_id` 和 `gt[N,12,2]`。ID顺序不一致直接拒绝，不静默重排。

`prepare`继承原split，生成 `X_train/GT_train/X_val/GT_val/statistics/INPUT_AUDIT`；不会重抽划分。训练统计包括显式train IDs；checkpoint绑定split SHA、准备阶段SHA、统计SHA、标签SHA、科学源SHA、完整训练合同与初始化/optimizer/RNG。

`masks`不接受GT参数的业务调用；`labels`必须先通过掩码完整封存检查。模型forward仅接收数值特征；动作解码和安全检查仅接收past、native和参数。group/track/scene只用于划分与报告。

held入口以人口数量／change数量做额外防误标检查，避免把hold误填为train：train3103/721、val993/256、hold3825/1023、inner1856/497。**计数并非身份认证**；正式执行仍必须核准并锁定每一人口的准确IDs与输入SHA。

## 4. 安全的检查命令

从项目根运行，下列命令只做合成测试或显示帮助，不读取真实实验人口：

```powershell
$obsPython = Join-Path (Get-Location) '.venv-governance/Scripts/python.exe'
$obsCode = Join-Path (Get-Location) 'docs/ENDPOINT_TEMPORAL_OBSERVABILITY_DESIGN_20261028/code'
& $obsPython -B "$obsCode/test_core.py"
& $obsPython -B "$obsCode/run_stage.py" --help
& $obsPython -B "$obsCode/compare_models.py" --help
& $obsPython -B "$obsCode/combine_saved.py" --help
& $obsPython -B "$obsCode/freeze_sources.py" --help
```

所有业务入口默认PLAN_ONLY，只有显式 `--run` 才实施。以下展示命令形状，`<...>` 是未来实际执行时必须解析的路径或实测值，不能当成已存在的收据，也不能在本轮直接粘贴启动：

```text
run_stage.py prepare --observed <X_opt> --gt <GT_opt> --split <原split>
  --out <新的prepared目录> --source-lock <本阶段源锁>
  --guard <resource_guard_v3.py> --inventory-bytes <实测E>

run_stage.py masks --observed <prepared/X_train.npz> --population train --branch END
  --out <新的END/train_masks> --source-lock <本阶段源锁>
  --guard <resource_guard_v3.py> --inventory-bytes <实测E>

run_stage.py labels --observed <prepared/X_train.npz> --gt <prepared/GT_train.npz>
  --population train --masks <END/train_masks> --out <新的END/train_labels>
  --source-lock <本阶段源锁> --guard <resource_guard_v3.py> --inventory-bytes <实测E>

run_stage.py fit --observed <prepared/X_train.npz> --population train --branch END
  --labels <END/train_labels> --statistics <prepared/statistics.npz>
  --arm H --seed 42 --device cuda:0 --preflight <实际完整预检收据>
  --out <新的END/H/42> --source-lock <正式训练源锁>
  --guard <resource_guard_v3.py> --inventory-bytes <与预检收据一致的实测E>

run_stage.py predict --observed <X_hold> --population hold --branch END
  --model-dir <END/H/42> --statistics <prepared/statistics.npz>
  --release <全部可执行final封存收据> --device cuda:0 --out <新的prediction目录>
  --source-lock <当前源锁> --guard <resource_guard_v3.py> --inventory-bytes <实测E>
```

这些多行示意不是直接可执行的PowerShell续行脚本。各阶段只处理一个分支／arm／seed；TIME和G须按原设计分别执行，不能把未执行的组合计为完成。

其他命令：

- `baseline --baseline identity`：原生identity参考，不产生模型seed复制。
- `baseline --baseline train_mean --train-observed ... --labels ...`：只从注册train教师取均值，再执行同一解码／guard。
- `evaluate --prediction-dir ... --gt ... --labels ...`：保存raw/guarded逐窗与分组指标；labels必须绑定同一人口、branch和GT SHA。
- `audit --prediction-dir ...`：当前仅第二套标量几何／动作／guard审核，**不是全量科学完成审核**。
- `compare_models.py --g <G42 G43 G44指标目录> --h <H42 H43 H44指标目录> ...`：读取guarded指标，不重新推理模型；输出CI但不自动判全部科学门。
- `combine_saved.py --endpoint-dir ... --time-dir ... --admission ...`：仅两H通过全部门后的同seed输出。不会选择GT最佳分支，不重新训练。

## 5. 源冻结与阶段复用

`freeze_sources.py`只接受明确的 `files: {path: sha256}` 父清单。相对路径相对显式 `--project-root`；先验证每个旧SHA，再扩展本轮18个源码及 `--extra` 中的资源适配器、设计、配置和真实输入。实际旧清单若是别的schema，应增加保留原digest的显式适配，而不是猜字段或重算覆盖旧SHA。

准备、标签与正式训练可以有递进源锁：每次继承前一锁并加入已经生成的不可变依赖，不覆盖已有digest；配对G/H正式训练必须使用同一最终训练源锁。runtime、日志和本轮输出目录不能加入会被自身更新的科学输入锁。

每个正常CLI阶段保存：

1. 数据／模型／报告文件。
2. `done.json`：输出文件SHA、代码digest及对应身份。
3. 最终源／资源检查收据。
4. `STAGE_COMPLETE.json`：绑定前两种收据，表示这个阶段输出封存完整，**不表示科学效果通过**。

只有 `done.json`、缺终态收据、存在 `FAILURE.json`、任何SHA或代码版本变化，均不能用作完成阶段复用。检查成功的已有阶段应由未来协调器跳过；当前CLI遇已存在输出目录直接拒绝，不自动覆盖、删除或重跑。

训练恢复需 `--resume <可信本地checkpoint> --resume-sha <显式SHA>`，输出仍选新目录。恢复核对数值输入digest、协议、统计、源、seed、device；不能将新标签或新源码套入旧optimizer继续。当前默认保存每epoch和final checkpoint，中途任意step恢复内核已在合成测试验证，但生产协调器尚未实现定时中途持久化策略，不能承诺任意宕机都无损恢复到最后一步。

## 6. 真实预检函数如何接入

`preflight_checks.py`提供函数，不自动伪造总体PASS：

```python
gradient_check(features8, targets8, device="cuda:0")
real_subset_resume(features513, targets513, validation12, validation_targets12,
                   contract, device="cuda:0", check=resource_guard.check)
tiny_fit(features_train, targets_train, sample_id_train,
         device="cuda:0", check=resource_guard.check)
```

features必须由本系列train-only统计构造，G/H分别测试；每个可执行分支各自核验真实ID。`gradient_check`使用固定非零末层的隔离double副本，不是生产模型；G被屏蔽的首层历史列零梯度是预期。

`real_subset_resume`两轮共34次更新，固定在第19次后序列化；同时核val12预测。函数无法自行证明传入数组确实是注册真实513/12，真实预检协调器必须提供ID和源证据。

`tiny_fit`仅按原合同从train哈希选64活跃窗+64其他窗、固定256更新；不足则STOP，不从val/hold补。当前返回活跃候选MSE下降与所选IDs，**尚需补教师动作命中、完整J-regret及全体MSE报告**，不能以 `PASS_TINY_MSE_ONLY` 冒完整预检通过。

正式fit要求收据状态 `PASS_REAL_DATA_GPU_RESUME_AND_TINY`、全部检查项、准确binding、device和实测inventory一致。该收据必须由真实检查结果与SHA证据产生，不是复制本文件的名称即可视为验证。本轮没有生成该收据。

## 7. 保持不变的科学细节

- END半径0.4m的49动作；TIME固定441网格及第三速率约束，零段长只identity。不是重做旧GT命中oracle。
- 先最近几何合法动作、再一次guard；不在失败后搜索第二安全动作，不按GT选择回退。
- 五motion+junction均literal `<= Native*1.01`；复算 `2e-12` 只比较算术，不加到科学门上。
- G/H共用标准化和初始化、同批序；G标准化后历史列置0。统计不使用val/hold/inner。
- 正式8epoch、batch32、AdamW .001/ .0001、clip5、无early stop；12个小模型是上限，不是必跑数量。
- 参数MSE与guarded J-regret分开；raw动作不属于safe集合时regret可以负，不能混入safe解释。
- 20条候选不是独立统计样本；三seed不扩成三倍窗口；六比较家族不因某分支STOP而缩小。
- P3只读已执行、已guard的同seed参数；在原Native上TIME后加END，再整体检查，不把改过的路径送入未训练过的TIME模型。

## 8. 正式运行前仍缺什么

| 环节 | 当前准确状态 | 还需完成 |
|---|---|---|
| 核心几何／训练／指标 | 已实现，合成CPU测试通过 | 真实train固定128窗口的独立全动作／教师标签检查、所有数据口径桥接 |
| 真实GPU与恢复预检 | 函数已有，本轮未执行 | G/H各分支实际GPU、配对初始化、真实513/12、tiny完整指标、吞吐及资源inventory |
| P0完整报告 | 保存safe/legal/逐项失败/J矩阵与机会 | 多道人口过滤贡献、歧义分层、全部源与实际ID审核报告 |
| 全量完成审核 | 标量几何／guard第二实现已有 | 独立复算全部质量、SciPy聚簇、pooled运动量、教师选择／歧义、CI与全部模型manifest的总协调器 |
| 全final放行／P3准入 | 消费收据并拒绝缺失 | 根据真实六或十二个final及STOP证据生成release；根据全部统计与保护门生成admission，不能手填PASS |
| P3最终评价 | 固定组合核心已实现 | 四端口完整指标／交互与两比较CI的结果协调器；目前普通单分支evaluate不能直接代替它 |
| 必须分层报告 | 基础overall/stable/transition/change已实现 | 原成功／原失败、教师活跃、scene、精细模式／可达性等描述，以及全train/val表 |
| 成本与独立worker | 没有启动，也没有生产启动器 | 精确资源预检、端到端成本拆分、sole OS worker、startup/half/completion流程接线 |

因此本轮可以交付“核心代码已补充且有合成测试”，不能交付“整套生产预检通过”或“科学实验完成”。下一次若明确执行，应先补完这些生产适配和证据链，再运行P0；P0不通过就保留停止原因，不继续训练凑结果。

所有旧模型、输入、科学源、失败、PAUSED任务均未更改；没有删除文件或恢复旧自动任务。
