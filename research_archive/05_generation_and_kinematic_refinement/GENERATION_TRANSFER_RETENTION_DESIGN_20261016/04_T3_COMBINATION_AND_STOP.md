# T3：只在两项独立信号成立后检验组合

状态：CONDITIONAL_DESIGN_ONLY；不是已经批准启动的worker。

## 1. 进入规则

只有T1的RETENTION_SIGNAL与T2的TRANSFER_SIGNAL同时满足，且无未解决source/ID/数值/资源障碍，才建议实施D的3个fresh runs。两步的CAL都只posthoc，不能据CAL筛选组合。

如果任一不满足，本系列停在A/B/C九runs，写整体分析。CI宽记INCONCLUSIVE，不通过追加seed或放宽门“补出显著”；增加数据/重复应另立研究，而非本轮隐式扩张。

机制阳性但motion未通过，可以做这一次已定义的D来检验组合是否自然改善代价；只能称机制组合实验。若D仍motion失败，结束本轮；不自动第四阶段投影/物理loss/限幅/scorer。

## 2. D与完整2×2结构

D=速度输入(C)+保留目标(B)。架构、输出核、初始化、数据、base loss与训练预算一概不变。D三个seed从fresh paired init训练，不加载B/C checkpoint。

对每个预注册指标M，报告：

\[
E_{keep|position}=M_B-M_A,\quad E_{repr|base}=M_C-M_A,
\]
\[
E_{keep|velocity}=M_D-M_C,\quad E_{repr|keep}=M_D-M_B,
\]
\[
I=M_D-M_C-M_B+M_A.
\]

coverage/rescue的I>0是正互补方向；lost/误差的I<0是正互补方向。正点估计但CI跨0只称“未确定交互”，不能宣传协同。D−A阳性也不自动表示D优于B和C。

## 3. 组合判定

**COMBINED_MECHANISM_SIGNAL**：

- D相对A outer change coverage每seed不下降，平均≥+2pp，family24 CI下界>0；
- D相对C lost平均下降，family24 CI上界<0；
- D相对B rescue平均增加，family24 CI下界>0；
- D的fixed-j0 ΔJ相比A改善，不靠ALL20平均替代；全部其他群体反例与motion如实报告。

即便以上成立，最终能否升级仍按共同协议重大成功全门。不把3pp lost改善或2pp coverage机制门当成生产标准。

## 4. 最终决策表

| 结果 | 应有结论 | 不应采取的动作 |
|---|---|---|
| B有效、C无效 | 监督可以保留，所选运动坐标没有迁移证据 | 自动堆MoE/新scorer或换十种feature |
| C有效、B无效 | 所选表示有迁移信号，当前keep不能兼顾探索 | 自动扫keep权重直到满意 |
| B/C都阴性或不确定 | 在这个有限预算下，两项具体干预未解决问题 | 将阴性当“行人不可预测/模型不可能理解” |
| D覆盖有效但加速度失败 | 准确率与运动代价仍冲突 | 降低1%物理门或直接进入scorer |
| D近native、lost低但rescue消失 | 保守退化，不是生成能力突破 | 把“安全”当“高质量生成成功” |
| D全面通过 | 有限数据上的候选生成改进值得独立迁移验证 | 宣称独立泛化、心理意图或SOTA |

## 5. 对更大分支的边界

这轮是在已有表征和候选支撑内测试“保留与迁移”，不是继续无穷小修。若A/B/C或D没有明确进展，下一次讨论应上升到**冻结P2/既有native支撑是否足够**这一更大问题，并综合之前G2/M2/D0/A0/S0/U0及R1–R5反证；不能自动换名字继续本轮的权重、步长、slot或encoder宽度变体。

更换/训练旧encoder、decoder、原scorer或使用新数据属于新的研究范围，本文不执行、不把历史授权当当前恢复命令。若全部门通过，下一步也应是独立迁移验证设计，而非直接宣称运动理解。

## 6. 最终报告必须包含

四cell的逐seed/逐split/逐group完整表；未执行D则明确NOT_TRIGGERED及哪个门失败。报告训练内改善、outer迁移、CAL反例、lost/rescue、关键slot与ALL20差异、全部五motion、计算审核与统计限制；每个结论附对应文件/ID集合/分母。

“执行完成”“机制信号”“重大成功”三栏分开。旧baseline/goal不随任务完成而标成科学成功。最后保留失败来源、源码锁和未完成项，不删除证据。
