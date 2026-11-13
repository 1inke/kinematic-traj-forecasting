# F0 结果分析：执行与审核完成，未取得有效安全见证

结论：`SAFE_WITNESS_PRESENT=false`，`SAFE_OPPORTUNITY_UNRESOLVED`。这不是“原安全域为空”，也不是模型训练失败。按原双门合同，本系列 **G1 不触发**；独立 I1 信息读出实验不受 F0 阴性影响，继续其自己的节点与完整分析。

## 1. 执行与审计事实

- 唯一生产 worker30780、exit 双 SUCCESS，2026-10-17T19:08:26.258142+09:00 完成；从 2026-10-17T18:57:10.707946+09:00 起实际 **11分15.550196秒**。没有因解除预算而减数据、slots或审核；没有真实题 walltime截断。
- 唯一预计半程检查 2026-10-17T19:15:25.8524978+09:00（原定19:14:54.817HK，约迟31秒）发现已经完成；本轮不再查已完成进程/task，也不重跑任何求解。
- 固定48个FIT change窗口、48个target tracks：32原生失败各20目标slot，**640次尝试全部完成**；另16原成功固定identity。不是640个独立窗口，没有训练seed/optimizer，也不增正式训练计数。
- driver `RESULT PASS_DIAGNOSTIC_ONLY_F0_SAFE_REACHABILITY`；原ARITHMETIC_AUDIT全640保存记录及完整FIT原质量/完整GT-free K20聚簇/五运动学通过；5732source初末SHA、1287完成输出SHA、PRESERVATION/RESOURCE通过。
- 追加CPU保存复核：[completion_review/REVIEW.json](completion_review/REVIEW.json)，全640 JSON+NPZ、原始primal→固定收缩atanh float32 exact、保存actualcontrols→独立递推、完整FIT原SciPy complete-link及全部逐slot质量/四组物理量重算。最大算术差 **4.163336342344337e-16**；全部源与完成输出SHA再次前后核验。无Torch、模型forward、optimizer、求解器或重训。这是同助手的第二保存算术实现，不是外部独立评审。
- 生产资源：parent Windows retained private峰2671677440bytes（2.488GiB）、真实owned solver树峰215158784bytes（约205.19MiB），两个峰不一定同时；观测aggregate currentprivate最大2166300672bytes。可用physical/commit最低12168695808/11515097088bytes，任务需求1.3规则通过；不用预检峰替代生产峰。
- 依据最新授权：隔离环境依赖已锁，30min及真实题2sec运行门已撤销；本轮的科学失败不是依赖或预算阻碍。solver仍固定Clarabel0.11.1/maxiter200/tol1e-9，无换solver/容差/权重搜索。

## 2. 640次尝试到底发生了什么

| 求解器标签 | 次数 | 可作何解释 |
|---|---:|---|
| infeasible | 616 | 该数值求解器在该凸子问题下的标签，不是完整pooled安全域不可行证明 |
| infeasible_inaccurate | 2 | 未得到可靠见证，不作为严格不可行证明 |
| optimal | 18（涉及14个窗口） | 只是求解器数值标签，仍须原始盘约束和实际精度完整审核 |
| SOLVER_EXCEPTION | 4 | 记录了CVXPY/Clarabel SolverError，不能算成功/不可行；没有自动换solver重试 |

4个异常的request为0:7、8:11、20:9、28:15，完整trace保存在原attempt JSON及补充REVIEW.json。driver正常保存这些未解决状态，因此worker SUCCESS不等于640题全部成功求解。

18个optimal标签中：

1. **14个控制primal norm>1**，超界范围 5.866196418e-12 至 2.214519901e-8。即使很小，原合同不允许用solver容差作为科学epsilon，故原样拒绝，不擅自归一化修成“成功”。
2. 其余 **4个** 进行了唯一固定收缩 `1−2^-20`、atanh→raw float32、原U0实际GPU tanh/normalize/float64积分重构。**四个的全部五项window运动学均通过**，但同slot ADE或FDE略超严格质量门，因此也不接纳。
3. 没有有效窗口，最低physical slot规则的chosen集合为空，完整FIT portfolio全部保持native bitwise exact。

### 4个实际重构案例（slot为0-based原physical编号）

| sample ID | slot | 实际ADE | 实际FDE | 失败门 | 运动学 |
|---|---:|---:|---:|---|---|
| 22199 | 3 | 0.127833393414 | 0.400000036654 | TARGET_SAME_SLOT_QUALITY | 全5项通过 |
| 11234 | 19 | 0.200000054571 | 0.400000131493 | TARGET_SAME_SLOT_QUALITY | 全5项通过 |
| 22731 | 15 | 0.200000029768 | 0.348552113319 | TARGET_SAME_SLOT_QUALITY | 全5项通过 |
| 1147 | 14 | 0.200000014724 | 0.218797692307 | TARGET_SAME_SLOT_QUALITY | 全5项通过 |

质量越界约1.47e-8至1.31e-7m；不能四舍五入显示成0.2/0.4后宣布成功。sample1147/slot14的原连续primal在保存的凸约束残差检查中全部≤0，但既定收缩/float32生产精度重构后ADE=0.20000001472369458，仍不构成实际可实现见证。这个反例尤其阻止把结果解释成“连续安全机会也不存在”。

这些4例全20槽都发生修正。11234/slot19与22731/slot15目标候选accelmean分别增加0.00542206、0.00561530，其他19槽Δaccelmean之和分别−0.05113579、−0.04484506；集合内补偿确实出现，但这些仍是**被拒绝尝试**，不能列入rescue或可部署策略。

### 全部18个optimal标签的资格

| sample ID | slot | max norm(u)−1 | 处理 |
|---|---:|---:|---|
| 22199 | 3 | -4.910338e-9 | 重构后同slot质量未过 |
| 22057 | 7 | 8.527179e-12 | 盘约束超界，未重构 |
| 26182 | 6 | 2.760113e-9 | 盘约束超界，未重构 |
| 11514 | 4 | 5.672780e-10 | 盘约束超界，未重构 |
| 11514 | 14 | 2.626712e-9 | 盘约束超界，未重构 |
| 20240 | 2 | 9.117239e-9 | 盘约束超界，未重构 |
| 20240 | 10 | 7.359049e-9 | 盘约束超界，未重构 |
| 23344 | 0 | 4.412370e-9 | 盘约束超界，未重构 |
| 14247 | 15 | 1.361956e-8 | 盘约束超界，未重构 |
| 11234 | 19 | -1.079024e-9 | 重构后同slot质量未过 |
| 20901 | 14 | 1.961826e-8 | 盘约束超界，未重构 |
| 22731 | 15 | -8.692739e-10 | 重构后同slot质量未过 |
| 2415 | 7 | 2.509180e-10 | 盘约束超界，未重构 |
| 2998 | 16 | 4.432870e-9 | 盘约束超界，未重构 |
| 1147 | 14 | -6.568956e-11 | 重构后同slot质量未过 |
| 10242 | 1 | 2.214520e-8 | 盘约束超界，未重构 |
| 10242 | 2 | 3.352692e-9 | 盘约束超界，未重构 |
| 10242 | 7 | 5.866196e-12 | 盘约束超界，未重构 |

## 3. 完整FIT最终输出：identity，不是安全生成成功

| group | N | native good → portfolio good | coverage | rescue / lost | minADE / minFDE / minJ | 有效good簇 / distinct |
|---|---:|---:|---:|---:|---|---|
| overall | 18909 | 10858 → 10858 | 57.4224% | 0 / 0 | 0.172974383 / 0.299229614 / 0.977744399 | 0.791581 / 19.722090 |
| stable | 10767 | 7414 → 7414 | 68.8585% | 0 / 0 | 0.133224069 / 0.229178901 / 0.752513575 | 0.989133 / 19.646420 |
| transition | 3417 | 1507 → 1507 | 44.1030% | 0 / 0 | 0.209907729 / 0.364804515 / 1.189969991 | 0.570091 / 19.891133 |
| change | 4725 | 1937 → 1937 | 40.9947% | 0 / 0 | 0.236845363 / 0.411434181 / 1.337508510 | 0.501587 / 19.772275 |

独立minADE/minFDE不拼作同一条成功路径；完整合格始终要求同slot ADE≤.2且FDE≤.4。有效簇按原完整K20 complete-link，非复制候选凑数。所有变化为0来自拒绝后identity，不是学会了保护，也不是证明新修正能兼得准确和安全。

| group | speedp95 | accelmean | accelp95 | jerkmean | jerkp95 | 相对native变化 |
|---|---:|---:|---:|---:|---:|---|
| overall | 1.275891287085 | 0.088411347796 | 0.634256418807 | 0.486259952643 | 2.800147005682 | 0%（identity） |
| stable | 1.336931832904 | 0.090546167150 | 0.635265010161 | 0.498001240523 | 2.929186706335 | 0%（identity） |
| transition | 1.216045301276 | 0.085056449694 | 0.617340389982 | 0.467807807833 | 2.614963655207 | 0%（identity） |
| change | 1.094525467349 | 0.085972847654 | 0.644914828610 | 0.472848848347 | 2.702423755775 | 0%（identity） |

## 4. 32个失败窗口全保留

| sample ID | canonical track | scene | 20次尝试标签 | valid slots |
|---|---|---|---|---:|
| 4069 | crowds_zara02::69 | zara | infeasible:19; SOLVER_EXCEPTION:1 | 0 |
| 22199 | students003::75 | university | infeasible:19; optimal:1 | 0 |
| 2703 | crowds_zara01::111 | zara | infeasible:20 | 0 |
| 25632 | students003::118 | university | infeasible:20 | 0 |
| 22057 | students003::84 | university | infeasible:19; optimal:1 | 0 |
| 22011 | students003::80 | university | infeasible:20 | 0 |
| 7604 | crowds_zara03::26 | zara | infeasible:20 | 0 |
| 23508 | students003::229 | university | infeasible:20 | 0 |
| 26182 | students003::380 | university | infeasible:18; optimal:1; SOLVER_EXCEPTION:1 | 0 |
| 11514 | students001::249 | university | infeasible:18; optimal:2 | 0 |
| 20240 | students001::209 | university | infeasible:18; optimal:2 | 0 |
| 537 | biwi_hotel::199 | hotel | infeasible:20 | 0 |
| 3153 | crowds_zara02::20 | zara | infeasible:20 | 0 |
| 23344 | students003::47 | university | optimal:1; infeasible:19 | 0 |
| 14247 | students001::103 | university | infeasible:19; optimal:1 | 0 |
| 15924 | students001::117 | university | infeasible:20 | 0 |
| 11234 | students001::90 | university | infeasible:19; optimal:1 | 0 |
| 16766 | students001::120 | university | infeasible:20 | 0 |
| 2561 | crowds_zara01::95 | zara | infeasible:20 | 0 |
| 13633 | students001::365 | university | infeasible:20 | 0 |
| 2227 | crowds_zara01::80 | zara | infeasible:19; SOLVER_EXCEPTION:1 | 0 |
| 20901 | students003::14 | university | infeasible:19; optimal:1 | 0 |
| 22731 | students003::316 | university | infeasible:19; optimal:1 | 0 |
| 8139 | crowds_zara03::41 | zara | infeasible:20 | 0 |
| 2415 | crowds_zara01::89 | zara | infeasible:19; optimal:1 | 0 |
| 24653 | students003::66 | university | infeasible:20 | 0 |
| 21654 | students003::28 | university | infeasible:18; infeasible_inaccurate:2 | 0 |
| 2998 | crowds_zara02::10 | zara | infeasible:19; optimal:1 | 0 |
| 1088 | crowds_zara01::16 | zara | infeasible:19; SOLVER_EXCEPTION:1 | 0 |
| 20187 | students001::392 | university | infeasible:20 | 0 |
| 1147 | crowds_zara01::17 | zara | infeasible:19; optimal:1 | 0 |
| 10242 | students001::82 | university | infeasible:17; optimal:3 | 0 |

16个native success身份和保留证据见原[sample_registry.json](sample_registry.json)与补充REVIEW.json，全在分母内，不按质量或求解难度换例。

## 5. 解释与下一分支

本轮证据最支持：**当前固定搜索/数值见证提取没有建立合格安全正例，且optimal部分主要卡在边界资格，而非已重构案例的五运动学。**

不能推出：原pooled安全域为空、历史观测无用、时间耦合一定无效、共享head/原loss是唯一根因、当前方法可学或可部署。单窗口均值约束比原group-pooled门强，最小加速度目标也未穷尽通过p95的全部解；618个含inaccurate的infeasible标签加4数值异常不能把未搜索域排空。

按注册合同，F0门未成立，**不启动G1**，也不在看结果后调整epsilon、收缩量、slot、solver容差、目标权重或补样本来求PASS。I1本身仍是有效独立问题，保持其原监督/输入/数据，等它完整结束后再汇总“安全机会未确定×信息信号”的分支；本轮未查询或停止I1进程，也未改其跟进。

## 6. ARS反方复核（同助手inline，非外审）

**DA2 — PASS_WITH_LIMITATIONS。** 最强反驳：“0个成功只是边界解被数值实现拒绝，不能称物理不可达。”成立，特别是1147连续primal反例和四个actual motion正例。因此结论限定为本次固定算法未建立见证，不将数值标签提升成安全域证明；不因此放宽原literal科学门。

**DA3 — PASS_WITH_LIMITATIONS。** 最强反驳：“完整FIT motion通过和原成功全部保留，看起来是安全成功。”不成立，因为最终portfolio恰好identity、16成功人为固定；零rescue必须同时披露。48个条件抽样tracks也不足以外推FIT/outer成功率。G1双门不满足与I1是否有信息是两件独立的事。

原DA1见[DA1_REVIEW.json](DA1_REVIEW.json)。完整结果来源：[RESULT.json](RESULT.json)、[JOBS.json](JOBS.json)、[FULL_FIT_AUDIT.json](FULL_FIT_AUDIT.json)、[ARITHMETIC_AUDIT.json](ARITHMETIC_AUDIT.json)、[completion_review/ALL_CASES.json](completion_review/ALL_CASES.json)。AI辅助实现与同助手复核已明确披露；无外审、无新训练或二次求解。

