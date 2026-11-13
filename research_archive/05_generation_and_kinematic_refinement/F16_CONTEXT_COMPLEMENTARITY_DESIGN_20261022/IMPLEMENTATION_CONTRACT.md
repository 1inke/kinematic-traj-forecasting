# 实现合同与核心接口（设计，不是已验收代码）

以 [DESIGN.md](DESIGN.md) 为科学定义。本文件提供接口和核心伪代码，不包含可直接启动训练的launcher。

## 1. 新执行目录

未来执行只新增本设计目录下`execution_<实际日期>/`，不得覆盖任何旧E0/E1/证据修复文件。

```text
execution_<日期>/
  protocol.json                   # 实际路径/SHA/环境填齐并冻结
  source_manifest.json
  input_contract.json             # token producer、字段、坐标、身份、曝光
  source_audit/RESULT.json
  split_ids.npz
  statistics.npz
  initial_states/seed*.pt
  batch_orders/seed*.npz
  teachers/seed*.npz
  PREFLIGHT.json
  BENCHMARK.json
  runtime_state.json
  runs/A_seed42/ ...
    checkpoint.pt
    run_manifest.json
    history.json
    assignment_stats.json
    predictions_FIT.npz
    predictions_inner.npz
    metrics.json
    done.json
  completion_review/REVIEW.json
  RESULT_ANALYSIS.md
  DECISION.json
```

所有拟产物均未在本次文档任务中生成。科学成功、执行成功和保存审核分别记录，不用一个PASS混淆。

## 2. 输入合同

```python
# 伪代码：由已核实 loader 提供，而非猜缓存字段
past: float32[N,8,2]             # focal local meters, last point=0
tokens: float32[N,8,96]          # 同一冻结 producer 的焦点观测编码
gt: float32[N,12,2]             # 仅loss/评价读取
native: float64[N,20,12,2]      # 只教师构造/评价，不进student forward
ids: int64[N]                   # 明确逐行身份
track: str[N]                   # sequence + pedestrian

v_last = (past[:, -1] - past[:, -2]) / 0.4
x = concat([past.reshape(N,16), v_last,
            tokens[:,-1], tokens.mean(axis=1)], axis=1)
assert x.shape == (N,210)
mean = x[FIT].astype(float64).mean(axis=0)
std = x[FIT].astype(float64).std(axis=0, ddof=0)
std[std < 1e-6] = 1
# 只保存FIT统计；三臂三seed完全共享
```

完整原生与GT精度保留，训练输入转float32的操作显式记录。教师mask在float64原路径上计算，训练教师坐标转float32后不得重新判好/坏。统计以固定生产feature生成后再分块累积，避免把不同float32后端差异混成统计实现错误。

## 3. 公共模型接口

```python
class ContextK20(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean.float())
        self.register_buffer('std', std.float())
        self.trunk = nn.Sequential(
            nn.Linear(210,128), nn.GELU(),
            nn.Linear(128,128), nn.GELU())
        self.queries = nn.Parameter(torch.randn(20,16) * .02)
        self.fuse = nn.Sequential(
            nn.Linear(144,128), nn.GELU(),
            nn.Linear(128,128), nn.GELU())
        self.head = nn.Linear(128,24)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features):
        # 唯一forward参数，无GT/native/teacher/group/logit/ID
        h = self.trunk((features-self.mean)/self.std)
        h = h[:,None,:].expand(-1,20,-1)
        q = self.queries[None].expand(len(h),-1,-1)
        z = self.fuse(torch.cat([h,q],dim=-1))
        return self.head(z).reshape(-1,20,12,2)
```

预期可训参数82008，统计buffers不计参数。需要同时报告冻结编码器参数、端到端MAC/实际延迟；不能以82008声称整模更轻。使用缓存进行训练不等于部署不需要编码器，推理不需要native decoder或KIN教师。

初始输出全零不是native identity。不得在验收中将二者混同。A/B/E每seed加载同一个完整初始state，包括stats；旧E1权重不参与。

## 4. 教师与损失接口

```python
def make_teacher(native, kin, gt):
    ng = good_same_path(native, gt)   # float64 ADE<=.2 AND FDE<=.4
    kg = good_same_path(kin, gt)
    success = ng.any(axis=1)
    R_mask = ng & success[:,None]
    N_mask = kg & (~success[:,None])
    return native, R_mask, kin, N_mask

def teacher_sums(student, teacher, mask):
    # 有效teacher先取出或安全masked，padding必须有限
    d = torch.linalg.vector_norm(
        teacher[:,:,None] - student[:,None], dim=-1)
    nearest = (d.mean(-1) + .5*d[...,-1]).min(-1).values
    return (nearest*mask).sum(), mask.sum()

def auxiliary(student, R, rm, N, nm, arm):
    sr,nr = teacher_sums(student,R,rm)
    sn,nn = teacher_sums(student,N,nm)
    # 以下if为语义伪代码，生产须记录count且保持autograd连接
    zero = student.sum()*0
    if arm == 'B':
        return (sr+sn)/(nr+nn).clamp_min(1)
    if arm == 'E_bal':
        if nr>0 and nn>0: return .5*sr/nr + .5*sn/nn
        if nr>0: return sr/nr
        if nn>0: return sn/nn
        return zero
    if arm == 'A': return zero
    raise ValueError(arm)
```

实际训练tensor命名避免`nn`覆盖torch.nn模块。教师距离中teacher/学生维度分别是T和K，不混成同slot损失。有效teacher可能为空；空组既不能NaN，也不能被当作高完成率。

`base_loss`从旧实际E1公式实现：hardmin20(ADE+.5FDE)+.05*endpoint_linear_hinge(.3)。编写独立NumPy标量对照，不能直接import会写旧目录的训练入口。

### 代数测试

B与计数加权分解在同teacher下的标量/所有合法参数梯度应在预设数值精度内一致，含R/N都空、仅R、仅N、两组都有四种情况。E_bal在两组计数不等时一般不等价；构造不等计数的非退化fixture检查它确实产生预期权重变化。

梯度测试不能只用zero-head：初始trunk梯度为零可能是解析零初始化后果；分别测试初始head梯度，以及固定非零head/实际数步后所有应连通参数。记录hardmin与hinge活动面，不把tie切换当梯度错误。

## 5. 预检清单（全部在正式启动前）

1. source/ID/producer lineage与FIT-only统计审核；缓存token与实际冻结模型固定128窗重放≤2e-5。
2. forward无GT路径：固定观测，改变GT/group/logit/teacher元数据不得改变预测。
3. 输出finite [B,20,12,2]、参数82008、模型初始化与所有buffer配对精确一致。
4. 基础损失标量和全参数梯度与原公式对齐；教师空组、成员、归一化、teacher-slot/student-slot轴测试。
5. 实际固定GPU计算batch256，输出chunk仅存储分块；不同存储分块得到精确相同输出。不能仅为匹配旧批次数值临时放宽科学门。
6. 每个训练臂在真实FIT513与inner12子集做2epoch连续/中断恢复：全部参数、统计buffer、optimizer、TorchCPU/CUDA、loader、NumPy/Python RNG、history/assignment一致。只证明该子集恢复，不冒称完整FIT恢复已运行。
7. 保存/重载实际128窗输出，由独立NumPy/SciPy复算完整质量/聚簇/五motion/教师mask；身份计数精确，浮点保存算术容差2e-12（同float64运算定义）。失败查实现，不改科学阈值。
8. 完成run的source/output/checkpoint SHA兼容才skip；注入错误源、输出、恢复contract必须拒绝。
9. 当前source/旧artifact初末SHA、内存动态规则、磁盘估算、吞吐、完整审核开销均PASS。

本文件没有声称任何预检已经通过。通过后再写冻结protocol，不能把草案里的占位路径直接视为合格来源。

## 6. checkpoint绑定与防错误续跑

每个checkpoint至少包含：model全部params/buffers、optimizer、epoch/step、CPU/CUDA/loader/NumPy/Python RNG、batch-order SHA、完整history与assignment、初始state SHA、protocol/source/IDs/stats/teacher SHA、软件版本与device/dtype。

`done.json`只有完整训练、18/6份相应人口输出保存审核、checkpoint/source锁和阶段门完成后按层次写入；单个run完成与S1/S2整阶段完成分别记录。失败保留code/log/partial，不重训旧已完成模型。

## 7. 单位一致的统计实现

```python
# d[s,i] = E_bal覆盖 - B覆盖；同sample同pipeline
# 每次draw：共同track重数 w[i]，共同paired-pipeline索引 ss
valid = change_mask
per_pipeline = (d[:,valid]*w[valid]).sum(1) / w[valid].sum()
draw_effect = per_pipeline[ss].mean()
```

不能对每个seed独立抽track后当作配对；不能将三seed样本拼接成3倍独立窗口。bootstrap输出保存draw规则、RNG、track清单与拒绝draw数量，独立按track聚合和展开window两个实现核对。

## 8. 输出与报告模板

报告先写实际状态：SOURCE_BLOCKED / BASELINE_NOT_READY / NO_SAFE_INCREMENT / LOCAL_SIGNAL_ONLY。科学门未通过也要交付完整负结果。

随后必须给：

- A/B/E/native的全部seed、FIT/inner、四组，非仅最好seed/change；
- native→学生和A→E、B→E的rescue/lost与分母；
- native成功、失败可教、失败无teacher三个子集，不误称后者不可预测；
- 每槽WTA与teacher责任、ties、distinct与effective-good；
- 五运动学逐项绝对值/比例及零基值；
- primary与预设secondary family区间，安全失败与统计不确定分开；
- source/provenance、上游暴露、重复inner、3pipeline等限制；
- 实测训练/保存审核时间、冻结编码器+学生完整推理成本；
- 阶段门是否触发后续，未做哪些项和原因。

不得将F16上下文模型对旧history-only学生的差异作为纯蒸馏收益，也不把局部门通过当成论文创新已经完成。
