# 03｜P1：责任分配 × 非负责候选保护，四臂匹配实验

状态：DESIGN_ONLY。主问题不是“history够不够”，而是同样输入下如何学习候选集合而不破坏原支撑。

## 1. 两个因素、四个臂

| 臂 | 训练责任 owner | 非owner几何保护 | 用途 |
|---|---|---|---|
| A | 当前预测 Q 最小slot | 关闭 | 原 U0/T2-A 目标的 fresh matched control |
| B | 固定原生 Q 最小slot | 关闭 | 单独检验固定责任政策 |
| C | 当前预测 Q 最小slot | 开启 | 单独检验非负责候选保护 |
| D | 固定原生 Q 最小slot | 开启 | 预先指定的联合方案及唯一 P2 候选 |

四臂所有结构、输入、统计、初始化、顺序、优化器、epoch相同，见01。A需新训，不把旧不同生命周期/资源记录下的模型当唯一对照；旧模型保留作历史参照，不重启旧任务。

12模型=4臂×3seed；25epoch×12=300history行。保存48个科学population job=12模型×FIT/outer/inner/CAL；四个native reference单列，不冒充科学训练job。

## 2. 目标的完整定义

设原生路径 b[i,j]，预测 p[i,j]，GT y[i]，K=20。

```text
Q[i,j]   = ADE(p[i,j], y[i]) + .5 FDE(p[i,j], y[i])
Q0[i,j]  = ADE(b[i,j], y[i]) + .5 FDE(b[i,j], y[i])
a_dyn[i] = first_physical_argmin_j Q[i,j]        # detach索引
a_fix[i] = first_physical_argmin_j Q0[i,j]       # 原生冻结，FIT训练标签

L_task = mean_i Q[i,a[i]]

d_native[i,j] = mean_t norm2(p[i,j,t]−b[i,j,t])
                 + .5 norm2(p[i,j,12]−b[i,j,12])
L_anchor = sum_(i,j≠a[i]) d_native[i,j] / (B×19)

dist[i,j,k] = mean_t norm2(p[i,j,t]−p[i,k,t])
L_div = mean_(i, j<k) relu(.30−dist[i,j,k])^2    # 全190pairs

L = L_task + .05 L_div + λ L_anchor
λ=0(A/B)，λ=1(C/D)，一次固定，不扫描。
```

Q与J责任不混用，本实验没有改成joint max loss。所有候选并列用最低原physical slot；ReLU零点导数0，欧氏范数零点采用框架已验证的零次梯度，不隐加epsilon改变目标。

L_anchor与Q都以米计量；λ=1只是预注册的单位权重配方，并非“最优权重”或已知等梯度强度。L_div仍为原平方距离项，不因单位不同重新调权。其梯度可能与anchor冲突，必须记录，不能事后删除原diversity求效果。

## 3. “保护”究竟承诺什么

**没有硬保护保证。** 所有20个slot仍通过同一个共享head；非owner也受diversity、共享参数及其他窗口梯度影响；owner本身也可能原来已经成功。推理时不知道GT、没有owner输入，也不能“19条不动、只修1条”。

L_anchor只是学习时为非owner提供回到各自native的梯度，避免把它们全部当作应向同一GT收敛的路径。它不证明真实未来多模态分布、不保证每个slot语义固定。

固定a_fix可能锁住不可修复或错误原生模式，压制更可修复的别的slot。W0不负责替它选择最好可修复slot；若B/D失败，按失败报告，不转用GT可达性标签救场。

与旧T1的差别：T1只对native已成功窗口、至少一条原好路径的J退化加惩罚；C/D对每个窗口19个非owner路径做逐slot几何保留，含原生失败窗口。不是把T1 keep换名字，但共享映射导致的保留—救回折中依然可能发生。

## 4. 保存的责任与模式诊断

最终各population保存完整Q0、Q、J、owner、原/新good masks、全K20路径、控制、每slot native drift、原/新聚簇。由此复核：

- a_dyn与a_fix不一致率、Q并列率、每slot owner占比、native-owner最终质量。
- 非owner中原本成功路径的retained/lost；是否原owner被修好但其他原成功模式被破坏。
- native→new nearest support distance（meanpath/.10、endpoint/.20两个原始量及D），双向最近距离都报告，不能只报一边。
- distinct/effective-good、rescue/lost、五motion，按4组和3seed完整列出。

固定诊断FIT前256 IDs，epoch0/1/13/25 eval保存所有slot Q/owner/路径；记录相邻检查点责任改变率。它只是四个检查点间改变，**不能称全部训练step switch rate**。训练batch按实际forward记录动态/原生owner分歧和ties，保留分母与epoch汇总；不因诊断改变训练顺序。

同一固定诊断batch记录任务/anchor/diversity的output-coordinate梯度范数与cosine，注明不是parameter梯度干扰的完整证明，也不是有限AdamW更新因果分析。

## 5. 预先指定的比较

- B−A、D−C：固定责任的条件效应。
- C−A、D−B：非owner保护的条件效应。
- D−A：联合方案总效应。
- D−C−B+A：交互项，不要求两个因素效应相加。

主终点为outer change的coverage、lost rate、rescue rate、mean minJ；六比较×四终点=family24。主对比D−A不是事后从B/C/D挑最大。统计详见05。

分布/保护结果必须与outer overall/stable/transition、inner、CAL、FIT一起报告，但只有固定outer是primary。不能用CAL获益或训练loss下降替换主检验。

## 6. P1→P2前置门（不是重大突破门）

只有预指定D同时满足下列要求，才运行04：

1. D−A outer change coverage均值≥2个百分点，三个seed不下降，family24区间下界>0。
2. D−A mean minJ三个seed不恶化，family24区间上界<0；rescue率均值增加、lost率均值不增加。
3. D相对native：outer change每seed coverage不下降，coverage family3区间下界>0；outer各组旧成功lost率≤1%。
4. D的outer overall/stable/transition ADE/FDE/J均≤native的1.01；D全部4split×3seed×4group五pooled motion均通过。
5. 完整源/ID/输出/模型/统计审核通过，且不存在身份混入、评价分母或协议混杂。

这些门可能很难通过，阴性不证明数学不可能。门未过：完整报告A/B/C/D，不自动改λ、owner、训练时长或网络。若C好D差，只能提出新的独立复核建议，不能事后把C塞进本系列P2。
