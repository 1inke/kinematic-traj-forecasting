# 核心实现合同与预检清单

状态：DESIGN_ONLY。以下是接口与伪代码，不是已经运行或通过测试的实现。不得将代码块直接称为生产核心已完成。

## 1. 模块边界

```text
execution/
  protocol.json / source_manifest.json / split_manifest.json
  input_audit.py          # ID、时钟、原生一致性；不训练
  action_geometry.py     # END49、TIME441及确定性动作解码
  motion_guard.py        # 仅past/native/proposal；不读取GT
  build_teacher.py       # GT隔离标签阶段，只在指定人口运行
  features.py            # 仅past和当前candidate，train统计
  model.py               # 共享4770参数MLP
  train.py               # 唯一固定训练合同
  evaluate.py            # 所有final锁定后才读hold/inner GT
  combine_saved.py       # P3，只用已有参数与Native
  audit_saved.py         # 独立CPU保存算术复核
  labels/ checkpoints/ predictions/ metrics/ failures/
```

所有写入限定新执行根；旧科学文件、4411来源链、输入、模型、失败不改。`source_manifest.json`继承既有原digest并扩展本轮实际依赖；不能把已有SHA换成当前值绕过冲突，也不冒称核验全部历史文件。控制状态与日志不放入不可变科学源清单。

## 2. 数据结构和禁止流向

```python
ObservedBatch:
    sample_id        # 存储/审计使用，不喂模型
    past_local       # [B, 8, 2], 最后一点为0
    native           # [B, 20, 12, 2]

LabelBatch:          # 与ObservedBatch分文件/分阶段
    gt               # [B, 12, 2]
    teacher_param    # [B, 20, 2], normalized
    teacher_action_id
    identity_J, best_J, feasible_count, ties, ambiguity

Prediction:
    continuous_param # [B, 20, 2]
    raw_action_id, executed_action_id, rejection_reason
    raw_paths, guarded_paths
```

`forward()`只接受已标准化的past/candidate数值；`decode()`和`guard()`只接受past/native/预测参数及冻结动作常数。GT、group、logit、teacher_J、标签有效性、oracle原因不得进入上述调用。原group/scene/track只能用于标签后的统计、划分和归因。

验证GT替换/打乱后模型特征、输出动作和guard完全不变；不要用打乱GT后“损失当然变了”混淆这个检查。

## 3. END49精确定义

整数生成 `[(i,j): i,j=-4..4, i*i+j*j<=16]` 共49个；按i、j升序登记动作索引，identity另有优先级而不修改原索引。参数delta=.1×(i,j)，单位米。

对t=1..12：`END(p,delta)[t] = p[t] + (t/12)*delta`，p0=0。identity使用原数组拷贝/只读引用，不重新加0。检查末点差为delta；未来内部速度增量为delta/(12*.4)，二/三阶差分理论不变；junction仍可能改变。

网络输出u∈[-1,1]²，对所有49个动作比较 `||u-delta/0.4||²`；最近者tie顺序：identity、参数范数、原动作索引。不要先径向clip再暗中改变距离定义。最终偏移一定落在半径.4m的固定有限集合；不是连续回归参数直接写入路径。

旧END0.4仅在GT端点差≤.4时直接命中GT，否则identity；这里格点与GT无关，可选不完全命中的偏移。因此两者不是嵌套等价oracle，不将新结果称复现旧120救回。

## 4. TIME441精确定义

令p0=0；每段长度l_t=||p_t-p_{t-1}||，L=sum(l_t)，q_t=cumsum(l_t)/L。只在L>0且每段l_t>0时非identity动作合法，其他情况仅identity。禁止用小epsilon强行制造单调或删除静止窗口。

`w1=q4; w2=q8-q4; w3=1-q8`。

`a,b ∈ {0.75 + 0.025*i, i=0..20}`，用整数索引生成，第三倍率 `c=(1-a*w1-b*w2)/w3`，要求0.75≤c≤1.25；identity显式包含并精确原样返回。

```text
t<=4:  q'_t = a*q_t
4<t<=8: q'_t = a*w1 + b*(q_t-w1)
t>8: q'_t = a*w1 + b*w2 + c*(q_t-q8)
q'_12 = 1
```

在原空间折线 `(0,p1,...,p12)` 上按原累计弧长位置插值，末点直接赋p12。这是改变采样进度，不是新增空间曲线；分段倍率不保证速度/加速度连续。

网络输出u，比较每个几何合法动作 `||(a-1,b-1)/.25-u||²`；tie顺序identity、归一化参数范数、固定网格索引。仅生成选中动作，再执行guard。部署不生成全部441条轨迹、不读取GT寻找更好的动作。

参数标准化中的.4与.25为冻结动作尺度，非train拟合统计。不能跨END/TIME直接比较未归一的参数MSE。

## 5. 两层安全评价，不能混用

### 5.1 部署候选过滤（新增保守政策）

逐候选：v_t=(p_t-p_{t-1})/.4，a_t=(v_t-v_{t-1})/.4（t=2..12），jerk由未来速度二阶差分/.4²得到。分别取speed p95、accel mean/p95、jerk mean/p95，沿时间计算。每项proposal≤对应Native×1.01才通过。

junction=||v_future1-(past[-1]-past[-2])/.4||/.4，同样proposal≤Native×1.01。零Native阈值就是0；浮点负零归零只属表示等价，不改变非零门。identity永远精确通过。

检查失败，完整候选回退Native；不得以GT决定是否回退。保存每一项失败位，多个失败同时记，不用“首原因”掩盖其他项。

### 5.2 原科学结果门

按人口、group、seed，把全部候选/时间元素合并后再计算pooled五项；不是平均各窗p95。逐候选各自p95通过不保证pooled p95通过，仍必须完整核验。

另外保存全20逐候选通过的窗口比例、每项失败数量，以及junction人口mean/p95。junction为本系列额外门；不能宣称历史实验也曾通过它。

## 6. 标签生成与多解

```python
for window in train_or_val:
    for slot in range(20):
        candidates = enumerate_fixed_actions(native[slot])
        admissible = geometry_mask & motion_junction_mask  # 无GT
        admissible[identity] = True
        J = maximum(ADE(candidates, gt) / .2,
                    FDE(candidates, gt) / .4)
        teacher = argmin_with_ties(J, admissible,
                                  identity_first=True,
                                  smaller_parameter_norm=True,
                                  stable_action_id=True)
        save_params_and_ambiguity(teacher, J, admissible)
```

标签的GT用途是训练监督，不是新增输入，不把“oracle知道未来”伪装成历史推断。最小J身份优先会产生大量identity标签；必须报告活跃候选和活跃窗口的比例。若多个动作近似等效，参数MSE会惩罚同样有效的另一个答案，因此同时评价**动作效用regret**，不能只凭MSE否定模型。

定义部署动作regret=`J(executed_action)-J(best_safe_action)`；identity属于safe集合。只对同动作集合内端口比较。raw安全外动作的regret可能为负，另列raw，不与safe regret混表。所有候选先窗内平均，再按窗口统计；不能把20槽当20个独立样本。

## 7. 模型与训练伪代码

```python
# 40 -> 64 -> 32 -> 2, 4770 parameters
x = concat(flatten(past8), flatten(native_candidate12))
x = (x - train_mean) / train_std
if arm == 'G':
    x[..., :16] = 0       # 标准化后屏蔽，不输入伪物理零历史
u = tanh(MLP_GELU(x))
loss_per_candidate = mean((u - target_normalized)**2, dim=-1)
loss = mean(mean(loss_per_candidate, dim=slot), dim=window)
```

历史最后一点恒0等零方差列置std=1。所有20slot共享同一MLP，没有单独slot embedding。G历史列首层权重对应输入恒0，梯度为0是预期，不虚报所有权重都有非零梯度。

同分支G/H共享初始化、原batch ID顺序与标准化数值；不同分支fresh拟合。常量标签基线使用train标签均值，其后也做同样动作解码和guard，不能获得额外GT过滤。

## 8. 一次性预检，不拿预检当结果

1. 输入manifest与全部继承来源初末SHA；不加载旧模型；原Native所有quality/ID逐项复现。
2. END49准确计数、TIME441网格/identity、单位、左右手性、端点/静止/零段长/边界/并列fixture。
3. 两个互不调用的NumPy实现，在固定train哈希128窗上复核动作、合法性和标签；动作ID不一致不能只因指标接近就略过。
4. 可解释算术容差在运行前按float64/float32分别登记；只用于复算等价，不用于good/motion门。选槽边界必须原样记录，不临时放宽gate。
5. 实际GPU两arm：配对权重、有限输出、零初始化identity、激活通道梯度、屏蔽通道零梯度；固定非零参数对MSE梯度用独立公式或double有限差分核验。量化/guard不反传，不能伪称验证了其梯度。
6. 每个可执行分支两arm真实513 train窗口、真实val12窗，子集2epoch连续/中断复现全部参数/统计buffers/optimizer/TorchCPU-CUDA/loader/Python/NumPy RNG、批顺序、history；不是完整3103两轮。
7. 每分支一次H seed42 tiny拟合诊断：train哈希选64个含非identity教师动作的窗口+64个其余窗口，固定256更新，batch32。若某层不足，保留实际数量并标规模不足，不从val/hold补；该GT分层仅用于训练预检，绝不替代正式人口。报告全体和非identity候选MSE下降、教师动作命中与J-regret。若非identity MSE未较初始化下降90%，标拟合预检未通过，停止对应正式训练并解释，而非调LR/网络直到过。通过仅说明这个小训练子集可拟合。
8. 全train固定模型在无更新状态下，预热4批、计时32批；另计动作生成、GT标签、安全检查、完整聚簇、bootstrap、checkpoint和IO成本。验证坏source/ID/stats/output拒绝与completed SHA skip。

预检只能证明当前实现满足合同，不能作为“历史有信息”的证据。失败保留源/收据，基础设施等价修复另存版本；科学动作、监督、split的更改必须作为协议修订，不暗改源hash。

## 9. 资源、冻结与后续执行流程

沿用resource_guard_v3：E来自实际预检峰与预计最大载荷inventory，1.3是预警，不是机械杀健康进程的阈值；不恢复旧6GiB硬上限或旧30分钟一刀切。物理/commit准入不得重复预留已分配内存。无OS/pagefile更改、无用户程序kill、无预占内存。

只跑一个模型/一个标签任务；完整动作轨迹有界分块。提前估计全部保存数据和复算临时空间，不自动删除旧文件腾空间。必要依赖在未来获准执行时按来源/版本锁定准备；本DESIGN_ONLY交付不安装，不将其误写成永久禁止依赖。

每阶段科学源锁+全部预检通过后才独立唯一worker；一次startup、一次50%预计elapsed、一次合理completion核验，按真实阶段修正，不高频查旧PID。不创建这些任务于文档交付轮。

所有模型final权重/优化器RNG/manifest绑定sourceSHA、splitSHA、statsSHA、action grid/mask合同、seed、batch顺序、objective、epoch；checkpoint不能只存参数。独立CPU审核保存结果不等于第二次模型或梯度重放。
