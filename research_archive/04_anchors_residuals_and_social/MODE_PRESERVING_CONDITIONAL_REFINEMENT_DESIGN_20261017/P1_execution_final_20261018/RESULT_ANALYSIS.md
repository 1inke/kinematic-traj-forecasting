# P1 模式保持条件修正：完整结果分析

2026-10-18。执行与保存审核 **PASS**；预注册联合方案迁移门 **FAIL**；重大成功 **FAIL**；**P2 NOT_TRIGGERED**。本有限系列按协议结束，不继续λ/owner/宽度/LR/epoch扫描。旧baseline、historycycle11及未完成总goal不变。

## 一句话结论

**固定候选责任只有不确定的小幅收益；非owner几何保护在本配方下主要产生近identity解，保留了旧候选却基本消除了救回能力。** W0证明有限GT样本存在安全修正机会，P1没有把机会变成可迁移的学习结果。这是本方法的保留—可塑性折中证据，不是生成数学不可能、历史无信息或唯一sharedhead根因。

## 1. 真实执行事实

- 唯一生产18720/service2804，10:14:58.651415至11:05:22.616143 HK，worker与exit双SUCCESS，实际50分23.964728秒，早于原200.085分钟吞吐外推规划。原预估不是生产保证。
- 4臂×3seed42/43/44×25epoch：12fresh模型、300history行、22200真实optimizer steps；48科学population jobs+4native=52jobs；156路径/aux/metrics主文件，连同checkpoint/manifest/history等共301个唯一输出SHA。
- 全部7457源、旧模型与结果、当前模型和输出SHA初末不变，digest `012d13345375993b881741b79829917179bee36df60e49add29a03ba5d9be692`。原F16/decoder/scorer未加载前向或训练；只新25304 head。不是又做一次旧模型GPU冻结重放。
- ARITHMETIC_AUDIT PASS：全52保存同candidate质量/J与Q各单slot、原GTfree全K20 complete-link/labels、五motion、owner/drift/support、controls独立递推、checkpoint/history合同及完整100k bootstrap第二实现复核；789.529秒。不是所有模型/梯度再次回放，也非外审。
- 本轮completion_analysis/review_saved.py另做CPU保存aux计数/错误/救回丢失IDs、全部208组记录、300history目标恒等式、初始化/顺序配对、主CI点估计独立复算与7457源/301输出前后SHA核验，58.187秒PASS；**本补充没有再次重算全部SciPy聚簇/CI或模型前向**，该完整审核由上述生产audit提供。没有重训或改阈值。
- 生产private峰4751478784 bytes（约4.425GiB）、RSS峰3381133312；physical最低14105292800、commit最低9856577536 bytes。E5700000000、1.3E7410000000动态current/Windows-retained守卫PASS；不套预检峰、不机械6GiB。
- 两个预检父版本的Path类型错误、核源期间作者I/O脚本更新被拒绝等全部保留；最终生产代码固定后完整重预检，生产stderr0。实现失败不混为科学阴性，科学阴性也不当bug自动重训。
- 人类本次“分析结果”直接核验保存结果，未查询18720或任何旧完成PID/task；不把它写成原定11:55半程已执行。runtime先前RUNNING是未更新控制记录，双SUCCESS才是完成事实。

## 2. 主结果：outer change（N=253，68 target tracks）

原生同一路径ADE≤.2 **且**FDE≤.4有110/253，覆盖43.4783%；原生失败143。所有斜线依次seed42/43/44。rescue以143原失败为分母，lost以110原成功为分母；净coverage以253为分母，不能把两种conditional pp直接相加。

| 臂 | 方法 | good | rescue | lost | 平均覆盖 | outer change 五motion |
|---|---|---:|---:|---:|---:|---|
| A | 当前owner，无保护 | 107/105/113 | 15/11/19 | 18/16/16 | 42.8195% | FAIL |
| B | 固定native owner，无保护 | 110/106/118 | 12/10/19 | 12/14/11 | 44.0053% | FAIL |
| C | 当前owner，几何保护 | 110/110/110 | 0/0/0 | 0/0/0 | 43.4783% | PASS |
| D | 固定native owner，几何保护 | 110/110/110 | 0/0/0 | 0/0/0 | 43.4783% | PASS |

A已有15/11/19个救回，但同时丢18/16/16个原成功，因此只有seed44净增。B相对A每seed净多3/1/5个覆盖，但相对native是0/−4/+8，尚不稳定。C/D都不是“保留同时学会修复”：它们outer rescue=0、lost=0，coverage回到native。

### 三seed FIT / inner / CAL 全部change分母与反例

| 臂 | split | N | native good | good(42/43/44) | rescue | lost |
|---|---|---:|---:|---:|---:|---:|
| A | FIT | 4725 | 1937 | 2171/2197/2161 | 358/374/343 | 124/114/119 |
| A | outer | 253 | 110 | 107/105/113 | 15/11/19 | 18/16/16 |
| A | inner | 497 | 166 | 173/173/174 | 26/23/23 | 19/16/15 |
| A | original_CAL | 700 | 243 | 221/214/225 | 22/18/23 | 44/47/41 |
| B | FIT | 4725 | 1937 | 2155/2144/2161 | 290/280/298 | 72/73/74 |
| B | outer | 253 | 110 | 110/106/118 | 12/10/19 | 12/14/11 |
| B | inner | 497 | 166 | 168/167/177 | 21/18/22 | 19/17/11 |
| B | original_CAL | 700 | 243 | 230/234/235 | 16/15/20 | 29/24/28 |
| C | FIT | 4725 | 1937 | 1937/1937/1937 | 0/0/0 | 0/0/0 |
| C | outer | 253 | 110 | 110/110/110 | 0/0/0 | 0/0/0 |
| C | inner | 497 | 166 | 166/166/166 | 0/0/0 | 0/0/0 |
| C | original_CAL | 700 | 243 | 243/243/243 | 0/0/0 | 0/0/0 |
| D | FIT | 4725 | 1937 | 1938/1938/1937 | 1/1/0 | 0/0/0 |
| D | outer | 253 | 110 | 110/110/110 | 0/0/0 | 0/0/0 |
| D | inner | 497 | 166 | 166/166/166 | 0/0/0 | 0/0/0 |
| D | original_CAL | 700 | 243 | 243/243/243 | 0/0/0 | 0/0/0 |

A FIT覆盖增加234/260/224窗口；B增加218/207/224，说明在训练人口确实能改善部分完整候选，不应概括为“head完全不能学习”。A inner173/173/174、B168/167/177 vsnative166也是保留的正例。原CAL A221/214/225、B230/234/235都低于native243，迁移问题仍在。C全部FIT change与native1937相同；D只在FIT seed42/43各救回1个、seed44为0，不能算实质生成能力。其他overall/stable/transition完整数据在 [ALL_GROUPS.md](ALL_GROUPS.md)，不是只报告change。

## 3. 固定责任到底有用吗？

B−A outer change覆盖点估计+1.18577pp，三seed方向都正；但是预注册family24同时CI为[−4.41109,+6.92318]pp，普通95%也跨0。minJ点估计−.010394，family24 CI[−.041634,+.020375]也跨0。**只能说有方向性线索，不能确认固定责任产生可靠迁移收益。**

B−native覆盖仅+0.52701pp，family3 CI[−4.05797,+5.70288]pp；其minADE/FDE/J的family9全部跨0。B原成功lost12/14/11，仍远超1%门，运动学也不合格。固定owner不是已被验证的解决方案；但这些宽CI也不证明该政策所有条件下等价/无效。

### 完整family24：六比较×四终点

表内为点估计[Bonferroni同时CI]。前3列单位pp，J为绝对值差；所有100000draw有效，未补抽、未缩family。

| 对比 | coverage pp | lost/native-success pp | rescue/native-failure pp | mean minJ |
|---|---:|---:|---:|---:|
| B-A | 1.18577 [-4.41109, 6.92318] | -3.93939 [-14.69433, 6.81683] | -0.93240 [-6.75538, 5.26316] | -0.01039 [-0.04163, 0.02038] |
| C-A | 0.65876 [-6.21019, 9.03541] | -15.15152 [-28.94624, -5.16192] | -10.48951 [-20.00000, -3.67647] | -0.01811 [-0.05625, 0.01950] |
| D-A | 0.65876 [-6.21019, 9.03541] | -15.15152 [-28.94624, -5.16192] | -10.48951 [-20.00000, -3.67647] | -0.01811 [-0.05625, 0.01950] |
| D-C | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] | -0.00000 [-0.00001, 0.00001] |
| D-B | -0.52701 [-7.35931, 5.28634] | -11.21212 [-20.16302, -3.92157] | -9.55711 [-19.86680, -3.10559] | -0.00771 [-0.03435, 0.02356] |
| D-C-B+A | -1.18577 [-6.92318, 4.41109] | 3.93939 [-6.81683, 14.69433] | 0.93240 [-5.26316, 6.75538] | 0.01039 [-0.02037, 0.04164] |

C−A及D−A显著降低lost，**也显著降低rescue**。D−A lost −15.1515pp（CI约[−28.946,−5.162]）与rescue −10.4895pp（CI[−20.000,−3.676]）是不同条件分母，不能说净改善15.15−10.49=4.66pp；实际净coverage只有+.65876pp，CI[−6.21019,+9.03541]。

D−C在这批outer样本的coverage/lost/rescue完全相同、CI[0,0]；J只差约−5.8e-7。该有限样本的指标相同不等于两训练政策在所有情况下等价，尤其anchor近identity已使二者几乎没有可观察的修复动作。交互项无可靠正证据。

[STATISTICS.json](STATISTICS.json)保留全部family24及native coverage3/errors9，不以有利的普通95%替代。

## 4. 几何保护为何不能算成功？

C在outer change的平均位移约3.15–7.46微米，D约4.35–10.89微米；最大位移约2.42e-5至5.32e-5米。对比A平均约.056米、B约.043米，保护臂基本退到identity，**但不是bitwise完全不更新**。

这与λ=1逐slot非owner几何惩罚在当前共享head上的保守解一致。不能用“原好不丢、簇数保持、运动学安全”替代救回能力。也不能仅从这个结果断言λ的唯一因果、共享参数干扰或表示不足；有限优化、目标几何和候选可修复性仍是替代解释。原协议已固定一次λ，不据结果追加局部扫描。

固定FIT256的动态Qwinner相邻epoch0/1/13/25变化：A各段4–17个，B2–16个，C/D0–1个。它不是全部训练step换winner频率。训练最后epoch中dynamic/nativeowner分歧A约14.5–14.9%、B约7.1–7.7%；这与“频繁切换一定是唯一根因”的说法不同。加anchor后分歧近0也可能只是路径不动，不是更懂责任。

输出坐标task/anchor梯度cos=0是owner与nonowner直接坐标mask互斥的结构结果；两者仍能通过共享参数耦合，**不能拿坐标正交排除共享参数干扰**。这些norm/cos只是描述，不证明有限AdamW更新因果。完整owner分布、tie、非owner旧good丢失、support双向距离/聚簇和梯度记录见 [completion_analysis/REVIEW.json](completion_analysis/REVIEW.json)。

## 5. 准确性与运动学关系

- A、B各48个split×seed×group组合 **全部五项AND门失败**；并不是五项每项都坏。主outer change失败项均为accel_mean：A较native约+12.285%–12.523%，B约+9.818%–10.172%，均超过1%。
- C、D各48组合五项全部PASS，但靠的是近乎不修正。保护误差overall/stable/transition的失败组合计数A14、B6、C/D0；这也不能补偿coverage或rescue门。
- [ALL_MOTION.md](ALL_MOTION.md)列全部16组native绝对参考及192科学组合五项相对变化与literal verdict；完整绝对值在补充JSON，不用显示四舍五入值判PASS。dt=.4，路径motion而非residual smoothness。

因此本轮没有同时获得“新增完整覆盖 + 原成功保留 + 运动学保护”。B减少了A的一部分运动代价和丢失，但不足以合格；C/D安全且保留，但没有实质学习增量。

## 6. P2逐门判定

| 预注册检查 | 结果 |
|---|---|
| coverage_mean_ge2pp | FAIL |
| coverage_each_seed_nondecreasing | FAIL |
| coverage_family24_lower_positive | FAIL |
| J_each_seed_nonworse | PASS |
| J_family24_upper_negative | FAIL |
| rescue_mean_increases | FAIL |
| lost_mean_not_increases | PASS |
| D_native_coverage_each_seed_not_worse | PASS |
| D_native_coverage_family3_lower_positive | FAIL |
| D_outer_each_group_native_lost_le1pct | PASS |
| D_outer_protected_errors_pass | PASS |
| D_all4split_3seed_4group_five_motion_pass | PASS |
| complete_execution_audit | PASS |

D−A coverage平均+.65876pp<2pp，seed44下降3窗口；family24覆盖CI跨0；J虽每seed点估计更小，但CI上界正；rescue均值下降；D−native覆盖CI[0,0]不满足下界严格>0。这些是科学门未过，不是依赖、时间或资源阻碍。D的保护/motion通过不能绕过这六个失败项。**P2 NOT_TRIGGERED；重大门FAIL；有限系列在12模型/300epochs结束，不凑到18模型/450epochs。**

## 7. 与W0及历史证据合起来看

W0的13/32失败窗口严格安全GT见证、全FIT组合运动学通过，反驳“这批样本没有任何安全改善机会”的笼统说法；但其GT辅助、全20共同预算和margin选择不是P1可部署策略，不能推出固定nativeQ最近slot最可修复。

P1在同输入/域/优化预算下：A/B在FIT有改善、在outer不可靠且CAL差，C/D在FIT也几乎没改善。当前联合方案的问题不只是一句distribution shift：保护政策已经压掉了源域可塑性。另一方面，W0不意味着所有窗口都可安全学会，也不能把P1阴性归为唯一loss或head根因。

保留F0原accepted0、不把数值边界写成空安全域；保留I1均值读出优CV但候选塌缩、额外历史迁移未成立；保留旧T1成功保留—救回折中、T2 FIT和部分inner正例但迁移/motion门失败、旧softset/energy-mixture反证。新四臂不是用一个弱回归基线人为制造优势。

## 8. ARS inline反方复核（同助手，不是外审）

**DA1：** 承接源码锁定的DA1_REVIEW：固定GT最近native owner可能不可修复；soft anchor可能只压制动作。已用四臂匹配和预指定D门检验，不能假设方法必有效。

**DA2：PASS_WITH_LIMITATIONS。** 最大风险是把C/D的lost=0包装为模式保持成功，或只挑B三个正向seed的B−A点估计。原始rescue/lost/微米级位移、所有校正CI及CAL/运动学反例已并列，拒绝该选择性解释。源数据有学习正例，但迁移和安全联合门阴性。

**DA3：PASS_WITH_LIMITATIONS。** 最强反驳是“λ=1没有合理梯度平衡，因此这不是候选保护原理的最终检验”。接受其范围限制：当前只否定这套固定配方已经成功，不否定所有保护方法；不以此为由在本轮追加λ/owner扫描。3seed、68outerchange tracks、3scene/相关窗口、F16已见、target-disjoint非context-disjoint及outer/CAL重复探索，使bootstrap只是有限探索证据，不是独立wholemodel泛化、SOTA或心理意图证明。

## 9. 后续大分支（建议，不自动实施）

应先明确“怎样保留原支撑同时允许非零的、可迁移的candidate-specific修正”，再考虑复核；不能继续以lost下降代替生成收益。与W0对照的缺口在于**存在性→可学习选择/修正→保留和迁移**，不是已确定需要更大head、MoE、scorer或旧模块解冻。更广上游表示/输出域或概率目标需要新独立设计与权限。

本轮不启动新训练、不调系数、不进入P2。仅清理本次mode-preserving-p1一次性跟进；旧暂停监控、Windows已完成任务、模型和所有证据保留。原RESULT.manual_review_pending是科学计算时的历史字段，不静默修改；本RESULT_ANALYSIS与runtime ANALYZED记录本次完整解释。

