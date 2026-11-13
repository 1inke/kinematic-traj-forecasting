# 实现合同、验证清单与反方检查

状态：DESIGN_ONLY。以下代码是接口级设计，不是已测试实现；本目录没有训练脚本或launcher。

## 1. 推荐代码组织

未来执行子目录中分别建立：

```text
source_contract.py       # 显式允许的旧只读函数；禁止旧顶层目录写入
features.py              # position / velocity 两种固定输入
model.py                 # 同一个120→128→64→24 head与U0积分核
objectives.py            # 原L0和唯一keep项
preflight.py             # 数值、梯度、ID、冻结、实际resume
train.py                 # 只允许A/B/C/D与42/43/44固定配置
evaluate_saved.py        # 四split最终完整保存与独立算术
bootstrap.py             # 固定paired抽样与全部预注册family
report.py                # 只读保存结果，不能选超参或修改模型
```

强类型分离：`forward(p2, native)`不得接收GT/groups/IDs/logits；`loss(pred, native, gt)`可使用FIT GT；`evaluate(pred, native, gt, ids, groups, tracks)`只用于posthoc。统计、feature版本、kernel、精度边界均进入checkpoint manifest。

## 2. 输入变换伪代码

```python
def candidate_features(native, representation):
    # native: [B,20,12,2]，局部坐标末观测点为0
    p32 = native.to(torch.float32)
    if representation == "position":
        return p32.flatten(-2)
    if representation == "velocity":
        p64 = p32.to(torch.float64)
        origin = torch.zeros_like(p64[..., :1, :])
        previous = torch.cat([origin, p64[..., :-1, :]], dim=-2)
        return ((p64 - previous) / 0.4).to(torch.float32).flatten(-2)
    raise ValueError("Unregistered representation")

def forward(p2, native, feature_mean, feature_std, head, kernel, rep):
    cand = candidate_features(native, rep)
    x = torch.cat([p2[:, None].expand(-1, 20, -1), cand], dim=-1)
    x = ((x.double() - feature_mean) / feature_std).float()
    z = head(x).reshape(-1, 20, 12, 2)
    u = torch.tanh(z).double()
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1)
    r = torch.einsum("tj,bkjd->bktd", kernel, u)
    # base精度边界必须与U0 inputs()一致；不可与feature重建值混用
    return native.float().double() + r, z, u, r
```

评估中immutable native参考仍保存源缓存原数组；预检核实U0缓存本身的float32表示与生产base合同，若不一致必须明确记录已有cast差，不能以“identity”掩盖。不得擅改源缓存使测试通过。

## 3. 保留项伪代码

```python
def path_J(paths, gt):
    e = torch.linalg.vector_norm(paths.double() - gt.double()[:, None], dim=-1)
    a = e.mean(-1) / 0.2
    f = e[..., -1] / 0.4
    # stack.max固定精确tie时ADE-first，而非另换maximum梯度约定
    return torch.stack([a, f], dim=-1).max(dim=-1).values

def keep_loss(pred, native, gt):
    with torch.no_grad():
        j0 = path_J(native, gt)
        good0 = j0 <= 1.0
        valid = good0.any(dim=1)
        reference = j0.min(dim=1).values
    if not bool(valid.any()):
        return pred.sum() * 0.0
    j = path_J(pred[valid], gt[valid])
    j = j.masked_fill(~good0[valid], float("inf"))
    # 不对空G0执行inf-inf；min按原slot顺序，mask后仍保留原顺序
    best_old_support = j.min(dim=1).values
    loss = 0.2 * torch.relu(best_old_support - reference[valid])
    return loss.sum() / pred.shape[0]

# A/C: old_L0(pred, gt)
# B/D: old_L0(pred, gt) + keep_loss(pred, immutable_native, gt)
```

这是对FIT真实标签的监督，不是无GT正则。不能在outer/CAL反向传播，不能把保留代表/原good mask喂给forward。zero-init时keep不活跃是正常的，不能擅改margin/coeff求非零。

## 4. 实际预检必须覆盖

### 来源与身份

继承U1最终2164相关digest及其新增结果/人工分析来源，旧digest逐项比对不可覆盖；锁定U0模型科学函数和本系列新代码。2164只是相关文件，不声称全历史34k。所有原ID、split、track、group、slot、native、P2逐项对齐；共同FIT-only统计独立复算。新feature统计chunk256，对固定1033样本分块累计31/256对照同一生产feature，使用原累计容差，不混float32 backend误差。

### 数值与梯度

- 原L0生产/独立Torch/NumPy值和全部参数梯度对齐；A用固定非零相同权重逐输出对齐U0 coupled。历史A重复不视独立泛化。
- keep fixtures：空G0、一个/多个G0、原good改善/恶化、旧bad变好、恰好J=1、精确J组件tie/候选tie、zero-init、整batch无success、混合batch分母。原bad优于全部G0时，不得偷换为全K20 min保留。
- loss梯度固定double参数副本eps1e-4/1e-5，固定非零权重与数据；对min/max/ReLU/控制归一化活动面切换显式标记。若切换导致有限差分不能验证，先使用解析已知fixture核验子梯度，不能不断搜epsilon/样本直到PASS；实际非光滑点不冒称平滑FD证明。
- GPU8实际FIT batch256，所有合法head层分阶段正有限梯度；zero-last初始前层零梯度需按阶段解释。loss正有限、完整K20 finite、位置界、native identity、参数/布尔/stat/kernel buffer配对。
- 速度变换：constant/stop/reverse/turn fixtures，double逆变换误差≤2e-12；实际float32 feature量化用8ε32乘以0.4Σ|v_t|的逐坐标界另证，不与科学质量epsilon混用；若数值界不成立停止修正计算证明，不放宽coverage/motion门。
- 真实模型保存重读outer128：完整GT-free聚簇、五motion、J/coverage、controls独立重建、模型所有weights/buffers保存前后exact。只是预检，不是科学job。

### 恢复与资源

所有将执行cell都跑真实513FIT+12inner子集2epochs continuous/interrupted；全部参数/布尔/stat/kernel buffers、optimizer、Torch CPU/CUDA、loader、Python、NumPy RNG、history和新增diagnostics EXACT。inner仍只posthoc。不是完整18909两轮，不是假称分析skip等于optimizer恢复。

已完成run复用须SHA/科学合同兼容；故意坏source/output/恢复合同必须拒绝。资源用共同协议task1.3；benchmark不保存科学模型、不按效果选择。只预检、完整源锁、预算均通过后才考虑独立执行。

## 5. 输出与审核

每run最少保存：checkpoint.pt、run_manifest.json、statistics.npz、split_ids.npz、history.json、assignment_stats.json、metrics.json；每split保存native/refined20、raw/controls/residual、显式ID/group/track、GT来源SHA与done输出SHA。keep的FIT native mask/参考J可压缩保存，绑定ID；外部群体mask只posthoc。

全部训练结束的审核由CPU saved-data进行：独立重算四split完整samecandidate质量、全部聚簇、五motion、keep/Q/assignment标量、kernel重构、每run25条epoch与1850更新步合同（18909按256每epoch74步，25×74）、固定bootstrap draws、统计分母及全部源/输出SHA。

history为训练过程preupdate batch-n加权值；不能拿最终checkpoint一次全人口loss要求与它相等。最终人口loss独立保存，用正确分母重算；不声称CPU从最终轨迹复现了所有历史梯度或optimizer。

复用A只读一次已认证结果，不重复worker。只在worker和exit均SUCCESS、所有done及审计齐备时分析。源码锁后基础设施修复须新版本/恢复子目录，保留失败，不静默换hash。最终报告同助手反方检查，不冒称外审。

## 6. 三轮反方检查：设计阶段完成，执行审核未完成

### 检查1：问题与方法

问题：把“更多信息”与“更好映射”混为一谈会破坏可辨识性。

修订：T2使用可逆同维速度坐标，不加raw历史/新context/MoE。代价：只能回答重参数化，不回答观测信息充分性。结论边界已写入T2。

### 检查2：证据与反例

问题：U1救回率差CI跨0，不能据此认定修复能力全面失败；CAL保持旧成功的上界只对现有R成立。

修订：同时报告lost、rescue、完整coverage、fixed-slot与ALL20；T1必须不牺牲救回。保留inner阳性、outer第三seed等反例。未用CAL选择方案。

### 检查3：最终设计压力测试

问题：保留项可能让模型退回identity；速度表示可能只改变条件数；组合筛选与反复outer查看有选择偏差；五motion仍可能失败。

修订：预注册救回非劣、净coverage、固定family与三seed一致门；D为条件性探索而非独立确认。最多12runs，禁止追加小变体；机制与重大成功分开。未加入任意新的安全投影避免同时改变因素。

剩余限制：3seed统计能力有限，family24区间可能很宽；没有新holdout，强结论必须等待另行设计的独立验证。设计可交付不等于源码、GPU、资源或科学预检已通过。
