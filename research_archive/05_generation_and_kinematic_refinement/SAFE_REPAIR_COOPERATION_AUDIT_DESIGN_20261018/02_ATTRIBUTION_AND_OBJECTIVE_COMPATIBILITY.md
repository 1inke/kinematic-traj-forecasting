# 02｜预算归因与目标兼容性

状态：DESIGN_ONLY；没有执行下述W0反事实。使用01固定的18份保存方案，不增加求解、训练、调参。

## 1. 两个均值指标可以做精确预算分解

对accel_mean、jerk_mean分别定义每候选的时间均值m_k。令δ_k=m_k(J)−m_k(N)，目标slot为j：

```text
完整集合变化 ΔM_J = sum_k δ_k / 20
只改目标变化 ΔM_T = δ_j / 20
原生允许增量 H = .01 * M(N)
目标单独超预算 d = max(0, ΔM_T − H)
其他候选净补偿 c = −sum_{k≠j} δ_k / 20
其他候选正补偿 c_pos = sum_{k≠j} max(−δ_k, 0) / 20
其他候选反向代价 c_neg = sum_{k≠j} max(δ_k, 0) / 20
c = c_pos − c_neg
```

保存原值、差值、单位、门限及算术重构误差。J通过、T不通过时，若该均值是失败项，以上恒等式能定量解释这条保存解中的预算补偿。不能把净补偿说成所有19条都在帮助；它们可能正负相抵。

零原生均值直接使用绝对预算0，不除零、不加floor。算术重构容差与literal科学门分开：重构近似相等不能让原本超1%的指标变成通过。

**候选是同一行人的互斥未来，不是20个行人。** “协同”指集合评价预算的共同调整，不是证明人际物理互动。

## 2. 三个p95指标不能按候选相加

speed_p95、accel_p95、jerk_p95必须分别对每种配置的全部20×时间元素重新计算原线性插值分位数。

- 保存q95两侧排序统计量、原索引/slot/time及并列数量。
- 原生q95和1.01×原生q95以上的tail数量仅为描述；不能用超阈值计数代替精确p95门。
- 不能平均候选p95，也不能把19个单候选p95下降相加作预算解释。
- DROP_k失败：在其余修正固定时，该候选修正对这条J方案是局部必要的。
- ADD_k通过：保留这一个已保存帮助者，足以配合这条固定目标修正。
- 全部ADD失败：只排除这些19种单帮助者组合，不证明必须19条一起改。
- 全部DROP通过：不证明完全不需要其他修正；可能存在冗余补偿者。

不进行更高阶子集搜索，不把这些量称为Shapley值或最小协同集。

## 3. 质量、模式支撑和修正代价

每配置同时记录固定目标ADE/FDE/J、全集合good、rescue/lost、distinct及effective-good簇。完整候选合格始终是同一条ADE≤.2且FDE≤.4。

支撑变化使用原GT-free距离D=max(meanpath/.10,endpoint/.20)：

1. 每条native到修正集合最近D；记录D>1的原生支撑数量。
2. 每条修正路径到native最近D；记录新支撑距离。
3. 原SciPy complete-link cut1的全K20簇数与含good路径簇数。
4. 目标与非目标各自的位移、残差v/a/j；残差运动学与完整路径运动学分列。

T保持19条native精确不动；J/O未必保持。即使J安全，其他候选的预算补偿也可能移动或合并原生模式，不能仅凭运动学低就认定是好集合。

这些窗口原来都没有good路径，无法检验“原有good成功保留”。16个成功identity对照未被修正，它们不丢失是构造事实，不是保护能力实证。

## 4. P1损失兼容性：仅标量评价，不求梯度

参考实际P1 `objectives.py`，在同一float64保存路径上用NumPy独立重写。每份方案的N/J/T/O分别计算A/B/C/D，18×4×4=288行；不加载模型或optimizer。

```text
Q_k = mean_t ||p_kt − GT_t|| + .5 ||p_kT − GT_T||
a0 = argmin_k Q_k(native)                 # 固定owner，首物理slot破并列
aX = argmin_k Q_k(X)                      # 动态owner
task = Q_owner(X)
drift_k = mean_t ||X_kt − N_kt|| + .5 ||X_kT − N_kT||
anchor = sum_{k != owner} drift_k / 19
div = mean_{190 pairs} relu(.3 − mean_t ||X_kt − X_lt||)^2
L_A/B = task + .05 div
L_C/D = task + .05 div + anchor
```

A/C用动态owner，B/D用固定native owner。保存W0目标j、a0、aX、质量J最优slot、所有并列及每项ΔL。W0指定目标与Q-owner不一定相同，不能合并成一个“winner”。批量时anchor分母为B×19，**不是19倍惩罚**。

安全方案若ΔL>0，说明该方案在当前目标中没有净激励；不能推出不存在其他安全下降方向。ΔL<0也不说明共享head能学到、优化器能到达或能迁移。接近浮点零的差异明确标注，不拿它作实质机制证据。

### 共同修正分量的结构性检查

若20条路径共同加同一δ_t，令qδ=mean||δ_t||+.5||δ_T||。C/D的anchor=qδ；由三角不等式，动态或固定Q任务改善至多qδ；pairwise diversity不变。因此λ=1时共同位移不能降低C/D总损失。

只描述保存方案中的共同分量占比：

`eta = 20 * ||mean_k residual_k||_F² / sum_k ||residual_k||_F²`。

分母0则null。eta不是梯度耦合证明；不额外构造“去共同分量”新测试，不扫描λ，不将GT方案用于训练teacher或部署owner。

## 5. 核心输出

`case_attribution.json`每方案保存：目标单独门、完整门、预算分解、p95归因、DROP/ADD各19行、支撑变化、四配置损失表。另存756行配置表、288行objective表及完整FIT四portfolio表。每行含原sample_id、窗口index、目标物理slot、来源SHA与primary/secondary/negative标签。
