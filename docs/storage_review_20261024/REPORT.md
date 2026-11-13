# 当前项目存储检查与清理建议

检查日期：2026-10-24。范围：`<WORKSPACE_ROOT>`。本次没有删除、移动、压缩或重建任何既有项目文件；只新增本报告和两张小型清单，并维护项目记忆。

## 结论

C盘检查时剩余约 **16.32 GiB**。项目文件逻辑大小 **137.57 GiB**，共 **541,039** 个文件。按文件ID去除项目内部硬链接重复计数后约137.49 GiB；NTFS实际占用及最终释放量可能不同。扫描无访问错误，未发现需要跳过的重解析路径。这是本项目检查，未扫描C盘其他项目或系统目录。

最有价值的三步是：有其他磁盘时优先迁出AV2原始数据；逐项裁剪历史中间epoch；停用旧selector方向后归档其缓存。清普通Python缓存只能释放约0.13 GiB。

所有释放量都是候选上限，不能直接相加：重复文件、缓存和实验目录之间可能重叠。只在C盘内部换目录不会增加可用空间；迁移必须到其他盘或外部存储，核验副本后才考虑移除C盘源文件。本次未执行迁移。

## 优先评估的候选

|顺序|路径/类型|现有体积|建议与条件|
|---|---|---:|---|
|1|`av2_dataset/`|52.88 GiB|最大的迁出候选。train 47.00、val 5.88 GiB，共449,792文件。是原始轨迹Parquet和地图JSON，不是垃圾。当前近期主线是ETH/UCY，但AV2准备脚本仍默认引用此路径；若近期不做AV2，建议完整归档到其他盘并调整路径。|
|2|各实验的`epoch_*.pth/pt`|10.30 GiB|共21,056个按epoch编号的文件，包含部分smoke。值得逐轮裁剪，但必须保留报告实际选中的epoch、best/selected、final、必需resume及其元数据；检查清单是否依赖这些原路径。10.30是全部候选体积，不是已确认可删量。|
|3|`SELECTOR_RANKABILITY_SERIES/`|4.01 GiB|几乎全部来自旧latent/centered缓存。S2V2约1.85、S2V3约2.00、S2 probe约0.15 GiB。若不再做旧selector训练/回放，优先归档大缓存，保留源码、配置、报告、结果与选中权重。|
|4|`SELECTOR_BENCHMARK_SERIES/cache/`|1.885 GiB|`.pt`是缓存分片，不能一概当模型权重。三seed体积接近，但大文件内容比对未将它们认定为重复；不能仅凭名称、大小或seed目录相似合并。停用该方向后可迁出。|
|5|已核验内容重复的大文件|额外2.34 GiB|完整SHA-256确认77组，详见DUPLICATES.csv。保留一份可节省内容空间，但直接删路径会破坏脚本和历史SHA清单；适合后续设计保留路径的去重/统一缓存方案。尚未验证原地硬链接适用于全部文件，不建议批量直接删除。|
|6|各级`__pycache__`、`.pytest_cache`及根`.cache`|合计约0.126 GiB|通常可重建，优先级高但收益很小。根`.cache`也含小型smoke模型，若要求完整预检证据，应先单独保留该部分。|

## 大体积历史材料：适合归档，不能直接当冗余删除

|目录|GiB|主要内容与建议|
|---|---:|---|
|`MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017`|5.66|其中P1最终执行5.52，主要是A/B/C/D完整预测与诊断。保留研究证据，长期不用可整体迁出。|
|`MOTION_STATE_GOAL_PROCESS_DESIGN_20261014`|5.13|其中M2为4.26，含多个seed的FIT预测。是完整实验材料，不因科学门失败就删除。|
|`PED_MOTION_UNDERSTANDING_SERIES_20261009`|4.69|其中stage2原始输入干预2.66、v3 gating约0.86。适合历史归档，留存报告和复现索引。|
|`GENERATION_GROUPED_TRANSFER_SERIES_20261006`|4.25|约2.93为saved输出，约1.23为模型目录；其中中间epoch候选1.17。identity_audit/splits.npz仍是当前配置引用输入，不能整目录删除。|
|`F16_ENDPOINT_PRESERVING_ACCEL_PROJECTION_DIAGNOSIS_20261005`|3.43|主要是恢复执行的逐窗NPZ，多个FIT文件每个约0.49 GiB。模型小、诊断大，归档价值高。|
|`F16_MATCHED_FIT_CAL_TRANSFER_DIAGNOSIS_20261005`|3.37|主要是jobs预测输出。可长期归档，报告和源清单留项目内。|
|`GENERATION_TRANSFER_RETENTION_DESIGN_20261016`|3.13|含T1/T2执行与重复参考预测；优先使用重复文件清单评估。|
|`qoder_work`|5.22|近期研究主线。10月24日三方向执行2.12、E0E1证据修复1.33、context tail 1.07，建议目前保留。|

上表是大小排序后的人工语义核读，不表示已经完成每个目录的全依赖检查。旧目录可能被新实验当输入；迁出时应维护路径映射，不能只按日期筛选。

## 真实重复示例

- `GENERATION_TRANSFER_RETENTION_DESIGN_20261016/T1_execution_20261016/runs/A/seed42、43、44/FIT_paths.npz`分别与`GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015/runs/integrated_acceleration/`下对应seed文件字节相同。这三对的额外副本合计约0.694 GiB。
- `qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022/runs/A_seed*/predictions_FIT.npz`与context-tail中`A_replay_seed*`的对应文件一致。
- `AV2_SCALING_SERIES/prepared/train_d4_data.npy`与`train_d6_data.npy`一致；d3/d5也一致。路径语义不同，删除前应修改对应读取方式。
- 根目录与`kinematic-traj-forecasting/`内的部分S2_LATENT_PROBE缓存一致。这个嵌套仓库总共仅0.156 GiB，清它收益有限。

核验范围：非AV2目录、单文件至少8 MiB且存在同大小文件的266个候选。先比较头尾，再对161个候选完整SHA-256，确认77个重复组；文件ID用于避免把现有硬链接当额外物理副本。没有解压读取NPZ数组，没有进行模型评测或实验重跑，也没有全面核验小文件重复。

## 其他可评估项

- 路径含smoke的材料约0.974 GiB，但包含断点恢复一致性等预检证据。建议只去掉已确认可重建的中间输出，保留预检结论、源码、必要失败记录；不能按smoke关键词整批删。
- `CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/.venv-k0`约0.427 GiB；`MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/F0_execution_authorized_20261017/.solver_env`约0.253 GiB。不再使用时可重建，但删除会丢失当时精确运行环境，应先保留依赖版本/安装来源。
- 上述F0的wheelhouse约0.059 GiB，近期Q0隔离runtime_deps约0.078 GiB；体积小，后者是刚完成审核使用的环境，暂不优先动。
- `.git`仅0.291 GiB；Git报告46个临时垃圾对象共约196 KiB。清理价值极低，保留版本历史，不要手工删objects。
- 全部日志仅约0.0043 GiB，Markdown约0.0077 GiB；保留失败日志、报告和研究记忆的空间成本很低。
- `.pt/.pth/.ckpt`扩展名合计约17.40 GiB，但其中包含缓存分片和Python路径配置，不能将其整体视为可删模型。

## 当前应保留的关键内容

- `src/`、`config/`、`docs/`、`tests/`、`audit/`、`reviews/`、`.memory`、`.git`和报告/协议/源清单。
- `qoder/work/native_fit_only/`、`qoder_work/v1_1/`及近期qoder_work执行结果。
- `config/paths.example.yaml`指向的：`REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz`、`F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz`与`metadata.npz`、`GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz`、`GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz`。这些就是“较旧目录仍有当前作用”的明确例子。
- 当前baseline、冻结native、正式选中checkpoint、split/ID、唯一原始数据、失败与恢复来源证据。
- 大量数据、缓存和权重已被.gitignore忽略。存在Git仓库不等于这些文件有备份；删后不能依赖git恢复。

## 建议执行范围

若有其他盘，先将AV2完整迁出，C盘可用空间理论上可从16.32增至约69.20 GiB，足以明显缓解压力。其后逐项评估epoch裁剪及旧selector缓存，暂时保留近期qoder_work和正式证据。

若没有其他存储，先处理约0.13 GiB普通缓存，并建立“可裁剪epoch—保留选中模型—报告引用”的清单；不要把52.88 GiB唯一原始数据直接当垃圾删除。重复副本去重还需要保留路径与只读使用约束，不能把2.34 GiB当作立即安全可删清单。

附件：`TOP_DIRECTORIES.csv`为全部顶层目录体积；`DUPLICATES.csv`逐项列出77组重复文件路径、SHA-256与体积。本报告及附件本身很小，不包含删除脚本。


## 补充：前述重点之外的约68.49 GiB是什么

为了避免重叠，本补充先互斥扣除AV2原始数据52.87885、整个SELECTOR_RANKABILITY_SERIES 4.00973、SELECTOR_BENCHMARK_SERIES/cache 1.88498，以及其余目录编号epoch文件10.30054 GiB，剩余68.49241 GiB、69943个文件。2.34 GiB重复副本与这些分类相交，不能再直接从总量扣除。普通缓存仍包含在剩余部分。

剩余中，NPZ约57.63504 GiB（84.15%），PTH/PT约5.21618 GiB，NPY约2.39873 GiB，其他约3.24246 GiB。主要是历次完整预测、逐候选运动学诊断、native参考、teacher输出、中间控制量、正式权重与准备特征，不是60多GiB源码。

已只读检查代表性NPZ的ZIP成员目录：端点投影FIT文件约508 MiB，保存52个数组，含原/投影轨迹、位移增量、speed/accel/jerk；M2某FIT约390 MiB，同时保存paths、velocities、projected_a、null_term、minimum_norm、raw_b、native20等；P1某FIT约238 MiB，保存paths、correction、controls、native等。单个实验按臂×seed×FIT/inner/CAL保存多份，因此累计数GiB。没有解压全部数组或执行评测。

|扣除上述候选后仍在的目录|GiB|
|---|---:|
|`MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017`|5.660|
|`qoder_work`|5.223|
|`MOTION_STATE_GOAL_PROCESS_DESIGN_20261014`|5.125|
|`PED_MOTION_UNDERSTANDING_SERIES_20261009`|4.687|
|`F16_ENDPOINT_PRESERVING_ACCEL_PROJECTION_DIAGNOSIS_20261005`|3.430|
|`F16_MATCHED_FIT_CAL_TRANSFER_DIAGNOSIS_20261005`|3.374|
|`GENERATION_TRANSFER_RETENTION_DESIGN_20261016`|3.132|
|`GENERATION_GROUPED_TRANSFER_SERIES_20261006`|3.073|
|`GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015`|2.016|
|`MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016`|1.647|
|`FAR_GOAL_SHORT_PATH_DESIGN_20261019`|1.638|
|`CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018`|1.633|
|`SHARED_MAPPING_GENERALIZATION_DESIGN_20261015`|1.629|
|`GENERATION_BRANCH_SCREEN_20261014`|1.406|
|`MOTION_PATTERN_TRANSFER_DESIGN_20261015`|1.337|

完整剩余目录表见REMAINING_BREAKDOWN.csv。近期qoder_work建议保留；较早实验的大型预测和诊断可评估迁出归档。诊断中的部分速度/加速度等可能能由轨迹重算，但尚未逐数组证明，不能据此直接删字段或重写带SHA的原文件。
