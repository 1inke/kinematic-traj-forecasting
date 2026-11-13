# 00 数据与共同评价合同

状态：DESIGN_ONLY。所有后续执行目录必须冻结本合同；发现来源不符先报告，不能静默换成相近数据。

## 1. 固定数据和版本

项目根：`<WORKSPACE_ROOT>`。

主要输入：

- `REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz`：sample_id、native candidate与change_group。
- `F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz`：相同ID的future。
- `GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz`：原grouped target划分及track身份。
- `GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz`：已保存past；只读取合法观察量。
- `KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{42,43,44}.pt`：冻结的三个已有KIN模型，不训练。
- `UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_{FIT,outer}_seed*.npy`与新V1的inner缓存。
- `COMPLEMENTARY_POOL_VALIDATION_20261018/artifacts/v1/*_bundle.npz`、`hgb_seed*.joblib`、原始source/manifest及`results/v1/seed*_*.json`。

候选池物理索引固定：0…19为native，20…39为KIN；12未来步、二维、dt=0.4s。native/GT/过去坐标原点与单位必须对齐。最终K20索引互异，不复制候选凑数，不改变路径值。

FIT18909、outer1035、inner1856、original_CAL2879。复用V1实际保存的FIT_train/FIT_cal ID划分，不重新随机划分、按效果换seed或补删窗口。训练子集见K1，所有评价窗口保持完整。

FIT_cal是selector校准子集，不是original_CAL；KIN此前见过原FIT，故即使selector track-disjoint，也不等于generator未见。outer/F16/原CAL的历史使用全部披露。

### original_CAL缺口

先找已保存、ID可验证的KIN缓存与真实sequence+pedestrian身份。若仅缺KIN缓存，可在后续获准实施本设计时单独做冻结eval/no_grad推理并保存：这不是重新训练，也不是“纯CPU读缓存”。先核对权重/缓存输出与资源再运行。

不能把loader默认`?|−1`当成全体CAL同一真实track。身份无法恢复时仍可报告全量点估计与运动学，但track CI标不可用；不能伪造独立窗口CI。若缓存/单位/源绑定有实质错误，相关split为BLOCKED。其余split结果可完成但不能声称全迁移验证闭环。

## 2. 来源锁和ID审计

冻结实际文件SHA、ID顺序SHA、split集合SHA、候选dtype/shape/排序、模型SHA和feature定义。继承已有manifest所列digests，不覆盖旧值；新审核标“当前取证”，不冒称补做了历史训练时锁源。

KIN.npy缺嵌入ID时，核对实际生成源码与注册split排序，再写绑定sidecar；无法建立确定绑定则STOP_SOURCE_MAPPING。不以长度相等代替ID证据。

所有变动只在本系列新的执行子目录。旧源码、旧模型、原始标签/结果及失败记录只读。

## 3. 质量指标

对窗口i、候选j：

`A_ij = mean_t ||p_ijt − y_it||`；`F_ij = ||p_ijT − y_iT||`；

`J_ij = max(A_ij/0.2, F_ij/0.4)`；`g_ij = 1[J_ij <= 1]`。

集合S的coverage为`c_i(S)=max_{j∈S} g_ij`，`J_i(S)=min_{j∈S}J_ij`。分别报告minADE、minFDE，并报告minJ对应的单一完整路径及早/中/晚误差；不拼不同slot。

对native参照保存所有retained/rescued/lost/both_fail、union_good_but_selected_fail的ID。真正lost必须满足`native_good AND NOT selected_good`，不是“原生good候选被删”。

conditional lost分母为原生成功窗口数；conditional rescue分母为原生失败窗口数。分母0时标NA，不制造0/0改善。

阈值0.3/0.5只作固定敏感性，不取代0.2/0.4主标准。全K原SciPy complete-link距离`max(meanpath/.10, endpoint/.20)`、cut1，保存distinct/effective_good；有效好簇不是越多越好，不以簇数替代覆盖。

## 4. 五项运动学

以观察末位置原点追加到未来路径之前，`v=diff(p)/.4`、`a=diff(v)/.4`、`j=diff(a)/.4`。

每seed×split×overall/stable/transition/change，合并所有窗口、所有20候选和相应时间步后，计算speed p95、accel mean/p95、jerk mean/p95。分位数沿用已锁定的linear口径。

逐项literal `metric_new <= 1.01*metric_native`，不加epsilon/floor。native为0直接比较绝对值，不用inf比值误判0对0。窗口门与pooled门独立保存；窗口p95通过不保证组pooled p95通过，不能偷换。

## 5. 分开“路线筛查门”和“项目成功门”

### K0路线筛查目标W（本次新登记，不是旧major替代）

在一个完整split、一个seed上同时满足：

1. change覆盖较native增加至少**2个百分点**，按整数`good >= native_good + ceil(.02*N_change)`判定；outer即至少116/253。
2. overall/stable/transition覆盖不低于native；各组原生成功conditional lost≤1%。
3. change平均minADE、minFDE各改善至少5%；overall/stable/transition各自两项误差不超过native的1.01倍。
4. 四组平均minJ均不高于native；五项pooled运动学全通过。

W比“随便找到一个救回”有区分力，但GT oracle达到W仍不是学习成功。额外minJ与2pp属于本系列预注册筛查要求，失败必须注明具体是哪一项，不能回头换目标。

### K1/K2项目判定

保留原change ADE和FDE各≥5%改善、保护组≤1%损害、三seed一致、同路径coverage/成功保留、相关性审核和每seed每组五运动学门。至少需要change coverage正增益且校正CI支持，不能以overall单项覆盖阳性升级baseline。新W门不取消任何旧门；K1还报告是否达到W。

K0 oracle不作“统计显著的模型收益”解释。K1/K2的推断对比见各自文档，点估计取原始样本，不使用bootstrap抽样均值替代。

## 6. 数据角色与防泄漏

- GT/group只用于FIT监督、K0 oracle诊断及posthoc评价。部署函数参数列表不含GT/group/scene/track/ID。
- K0读取outer GT是离线可达性诊断；它不能产生传给K1的teacher动作、标签或样本筛选。K1只使用FIT_train自身的真值计算监督。
- K0结果仅决定GO/STOP分支，不能据outer修改K1特征、预算、网络、阈值、抽样或超参数；实施时这些合同应在K0前一起锁定。
- 两seed模型输出和同一窗口不是独立新样本；三seed并非充分的模型总体估计。
- 同为K20不等于同计算预算：仍生成40条并运行选择器，报告全部成本。
