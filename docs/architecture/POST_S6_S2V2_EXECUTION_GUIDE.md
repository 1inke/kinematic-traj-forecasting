# Post-S6 修改执行文档：结果可信度修复 + S2-v2 Context-Conditioned Latent

> Repository: `1inke/kinematic-traj-forecasting`  
> Current target commit: `3a10ae61ecdaa5c9ea2a9a5932d026b2ab447aec`  
> Scope: 修复 S6 系列中影响结论可信度的问题，并启动 S2-v2 Context-Conditioned Latent 系列。  
> 核心原则：**先修实验一致性，再扩展 latent；暂停继续堆叠 S6G/S6H 类 Local Reranker 调参。**

---

# 0. 当前必须接受的阶段性结论

当前最强 Macro ADE 仍为：

```text
Global Transformer Residual (S1A-v3)
ADE = 0.5077m
```

S6 系列：

```text
S6A best local        ≈ 0.5088m
S6B best              ≈ 0.5094m
S6C best              ≈ 0.5088m
S6D best gated        ≈ 0.5105m
S6E fixed-K3          ≈ 0.5094m
S6E adaptive-K        ≈ 0.5113m
```

因此当前不能再写：

```text
Local Reranking 已显著超越 Global Reranking
Adaptive-K 已达到 Pareto Optimal
Candidate Truncation 在 K>=5 时绝对不是瓶颈
Hard-case weighting 已被证明显著有效
Near-Oracle loss 已显著提升 accuracy
```

更准确的阶段结论应为：

> Local Top-K candidate pools retain substantial oracle headroom, but current deployable post-hoc feature families (G0/G1/G2/G4/GL) only exploit a small fraction of this headroom. Local reranking is competitive with, but does not yet consistently outperform, the strongest global residual selector.

---

# 1. 总体执行顺序

严格按以下顺序执行：

```text
P0-A  S6 Reproducibility Audit
P0-B  S6 Report Conclusion Fix
P0-C  Collision Metric Unification
P0-D  Strongest Global vs Local Re-comparison
        ↓
P1-A  S2-v2 L3 Variance Audit
P1-B  L4/L5 Hook Discovery
P1-C  Context-Conditioned Latent Extraction
P1-D  Global Latent Probe
P1-E  Local Latent Probe
        ↓
Decision Gate
        ↓
若 ADE < 0.500 → Tiny Latent Reranker
若 ADE >= 0.505 → Large Capacity Probe / Representation Limit
```

---

# 2. P0-A：S6 Reproducibility Audit

## 2.1 问题

当前同一配置：

```text
K5_tiny_set_attention_g0124_gl
```

在 S6A：

```text
ADE ≈ 0.5088m
```

在 S6B 的：

```text
B0_baseline
```

中变成：

```text
ADE ≈ 0.5172m
```

这不是可以忽略的随机误差。

主要来源：

```text
S6A outer neural training = 6 epochs
S6B B0 neural training    = 5 epochs

不同脚本仅在启动时 seed 一次
config/fold 之间 RNG state 持续变化
```

---

# 3. 新增 deterministic seed 工具

新增：

```text
SELECTOR_LOCAL_RERANK_SERIES/common/reproducibility.py
```

实现：

```python
from __future__ import annotations

import hashlib
import random
import numpy as np
import torch


def stable_seed(
    base_seed: int,
    *parts: str | int,
) -> int:
    raw = "::".join(
        [str(base_seed), *[str(x) for x in parts]]
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16
    ) % (2**31 - 1)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

# 4. 每次模型训练前重新 Seed

修改：

```text
S6A_LOCAL_K_SWEEP/run_s6a_local_k_sweep.py
S6B_HARDCASE_TRAINING/run_s6b_hardcase_training.py
S6C_NEAR_ORACLE_LOSS/run_s6c_near_oracle_loss.py
S6D_GATE_LOCAL/run_s6d_gate_local.py
S6E_ADAPTIVE_K/run_s6e_adaptive_k.py
```

每个 outer fold / config / stage 在模型创建前：

```python
train_seed = stable_seed(
    args.seed,
    cfg_id,
    test_fold,
    "outer"
)

set_deterministic_seed(
    train_seed
)
```

Inner calibration：

```python
inner_seed = stable_seed(
    args.seed,
    cfg_id,
    test_fold,
    inner_val_fold,
    "inner"
)

set_deterministic_seed(
    inner_seed
)
```

---

# 5. 统一 Neural Training Hyperparameters

新增：

```text
SELECTOR_LOCAL_RERANK_SERIES/common/training_config.py
```

例如：

```python
NEURAL_EPOCHS = 8
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
```

所有 S6A/S6B/S6C 共享。

禁止：

```text
S6A = 6 epochs
S6B = 5 epochs
S6C = 另一套 epochs
```

---

# 6. 统一 Neural Training Function

将训练逻辑统一移动到：

```text
common/local_models.py
```

提供：

```python
train_neural_local_ranker(
    model,
    x,
    ade,
    device,
    *,
    sample_weights=None,
    sample_mask=None,
    epochs=NEURAL_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
```

S6A：

```text
sample_weights=None
sample_mask=None
```

S6B B1：

```text
sample_weights=hard_weights
```

S6B B2：

```text
调用相同 train function
但按照 curriculum mask 分阶段
```

---

# 7. 新增 EXP-S6R：Reproduction Audit

新增目录：

```text
SELECTOR_LOCAL_RERANK_SERIES/
└── S6R_REPRO_AUDIT/
    ├── run_s6r_repro_audit.py
    └── results/
```

只验证以下 3 个代表配置：

```text
K3_linear_residual_g0124_gl
K5_tiny_set_attention_g0124_gl
K10_tiny_set_attention_g0124_gl
```

每个配置分别：

```text
S6A baseline implementation
S6B B0 implementation
```

要求完全使用：

```text
same seed
same epochs
same fold split
same alpha
same scaling
same feature set
```

---

# 8. S6R 成功条件

要求：

```text
Linear model:
|ADE_A - ADE_B| < 0.0001m

Neural model:
|ADE_A - ADE_B| < 0.0005m
```

即：

```text
< 0.5mm
```

如果不满足：

```text
STOP
```

不要继续解释 S6B/S6C 增益。

---

# 9. P0-B：修复 S6 报告结论

需要修改：

```text
SELECTOR_LOCAL_RERANK_SERIES/LOCAL_RERANK_SYNTHESIS_REPORT.md

S6F_LOCAL_ORACLE/results/RESULT_ANALYSIS_S6F.md
S6A_LOCAL_K_SWEEP/results/RESULT_ANALYSIS_S6A.md
S6B_HARDCASE_TRAINING/results/RESULT_ANALYSIS_S6B.md
S6C_NEAR_ORACLE_LOSS/results/RESULT_ANALYSIS_S6C.md
S6D_GATE_LOCAL/results/RESULT_ANALYSIS_S6D.md
S6E_ADAPTIVE_K/results/RESULT_ANALYSIS_S6E.md
```

并同步修改对应的 Markdown generator，避免手工改报告后下一次被覆盖。

---

# 10. S6F 结论修改

删除：

```text
Candidate truncation 在 K>=5 时绝对不是瓶颈
```

改为：

> Native Top-K pools preserve substantial oracle headroom. However, candidate truncation remains a meaningful ceiling, especially for K=3 and K=5. Top-10 provides substantially stronger oracle coverage than Top-5.

必须明确：

```text
Top-5 Exact Oracle Recall ≈ 60%
Top-10 Exact Oracle Recall ≈ 81~85%
```

---

# 11. S6A 结论修改

删除：

```text
Local paradigm 超越全局
```

改为：

> Local reranking reaches performance comparable to strong global residual reranking, but has not yet demonstrated a consistent Macro-ADE advantage over the strongest global Transformer residual baseline.

必须在报告中新增：

```text
Strongest Global Reference:
transformer_residual_g0124
ADE = 0.5077m
```

不要只和旧的：

```text
linear_residual_g012 ≈ 0.5098m
```

比较。

---

# 12. S6B 结论修改

当前允许写：

> Hard-case weighting and curriculum provide marginal and configuration-dependent changes.

不要写：

```text
Hard-case weighting 明显有效
```

尤其 linear deterministic configs：

```text
收益约 0~1mm
```

不足以支撑强结论。

---

# 13. S6C 结论修改

当前最佳：

```text
ADE ≈ 0.5088m
```

并没有突破 S6A 最好结果。

改为：

> Near-oracle-aware supervision remains competitive and can alter intervention behavior, but does not improve the current Macro-ADE frontier.

---

# 14. S6D 结论修改

删除：

```text
Local Reranking 始终优于 Global Reranking
```

实际：

```text
5%~30% budget:
Global better

50%~100%:
Local marginally better
```

改为：

> Global and local rerankers exhibit budget-dependent trade-offs. Local reranking does not consistently dominate global reranking under selective intervention.

---

# 15. S6E 结论修改

删除：

```text
Adaptive-K is Pareto optimal
```

实际：

```text
Fixed K3:
ADE = 0.5094
Avg K = 3.0

Adaptive-K:
ADE = 0.5113
Avg K = 6.7
```

改为：

> The current gate-score-based Adaptive-K policy does not yield a favorable accuracy-efficiency trade-off. Fixed K=3 dominates Adaptive-K in both Macro ADE and average candidate count.

S6E 当前结论：

```text
NEGATIVE RESULT
```

暂时停止进一步阈值微调。

---

# 16. P0-C：统一 Collision Metric

当前：

```text
Phase 1:
collision only on social-eligible samples

S6:
collision on all samples
```

因此：

```text
6.68%
vs
5.97%
```

不能直接比较。

---

# 17. 修改 local_metrics.py

函数新增：

```python
has_neighbors: np.ndarray | None = None
```

输出两套：

```text
collision_all
collision_social
```

实现：

```python
collision_all = float(
    np.mean(collision)
)

if has_neighbors is not None:
    social_mask = has_neighbors > 0

    if np.any(social_mask):
        collision_social = float(
            np.mean(
                collision[social_mask]
            )
        )
    else:
        collision_social = float("nan")
else:
    collision_social = float("nan")
```

---

# 18. Safety 主报告标准

以后：

```text
Collision_Social
```

作为主要 safety metric。

```text
Collision_All
```

作为 secondary。

保持和 Phase 1 一致。

---

# 19. P0-D：重新做 Strongest Global vs Local

新增：

```text
EXP-S6G_STRONG_BASELINE_COMPARISON
```

注意：

这里的 S6G 不是继续搜索新架构，只做公平 baseline 对比。

对比：

```text
Native

Global:
transformer_residual_g0124
linear_residual_g012

Local:
best deterministic local
best neural local

Gate + strongest global
Gate + best local
```

统一：

```text
same seed
same outer folds
same macro/micro
same collision metric
```

---

# 20. Strong Baseline 结果表

至少：

```text
Method
ADE Macro
ADE Micro
FDE Macro
Regret Macro
NearOracle05
Collision Social
Intervention Rate
Latency
Params
```

---

# 21. S6 最终停止条件

完成：

```text
S6R
+
S6G
```

后，如果：

```text
best local >= 0.505
```

则正式停止继续扩展：

```text
G0/G1/G2/G4/GL post-hoc local features
```

进入：

```text
S2-v2
```

---

# 22. P1-A：S2-v2A L3 Variance Audit

当前旧 S2 提取：

```text
decoder.mode_mlp output
L3
[B,20,64]
```

但尚未确认 L3 是不是 sample-conditioned。

新增：

```text
SELECTOR_RANKABILITY_SERIES/
└── S2V2_CONTEXT_LATENT/
    ├── run_s2v2a_l3_variance_audit.py
    ├── extract_context_latents.py
    ├── run_s2v2c_global_probe.py
    ├── run_s2v2d_local_probe.py
    └── results/
```

---

# 23. L3 Variance Audit 指标

对每个 mode：

```text
within-mode cross-sample variance
same-mode cross-sample cosine similarity
between-mode centroid distance
between-mode variance
```

计算：

$$
R_{mode/sample}
=
\frac{
Var_{between-mode}
}{
Var_{within-mode}
}
$$

如果：

```text
same-mode cosine > 0.95
within-mode variance 很低
```

则 L3 更接近：

```text
fixed / weakly-conditioned mode embedding
```

当前旧 S2 不能继续作为 latent upper-bound 证据。

---

# 24. L3 额外检查

对随机 sample A/B：

```text
cos(
L3[A,k],
L3[B,k]
)
```

与：

```text
cos(
L3[A,k],
L3[A,j]
)
```

比较。

输出：

```text
same_mode_cross_sample_cosine
different_mode_same_sample_cosine
within_mode_variance
between_mode_variance
```

---

# 25. P1-B：寻找 L4/L5

目标：

```text
L4:
mode-specific hidden after context fusion

L5:
confidence-head input / pre-logit hidden
```

要求：

```text
[B,20,D]
```

或：

```text
[B,N,20,D]
```

其中 focal agent 取：

```text
[:,0,:,:]
```

---

# 26. Agent 定位步骤

在 F16 / decoder forward 中搜索：

```text
mode_mlp
context
mode_query
confidence
score
logit
trajectory_head
output_head
decoder
cross_attention
```

重点画出：

```text
history/context representation
        ↓
mode representation
        ↓
context-mode fusion
        ↓
trajectory output
        ↓
confidence/logit output
```

必须先写：

```text
LATENT_HOOK_MAP.md
```

内容：

```text
tensor name
source file
source line/function
shape
before/after which operation
sample-conditioned? yes/no/uncertain
mode-conditioned? yes/no
used by trajectory head?
used by confidence head?
```

---

# 27. Hook 原则

禁止修改数值路径。

只：

```python
register_forward_hook
```

或：

```python
if debug_capture:
    cache[name] = tensor.detach()
```

必须验证：

```text
capture off
vs
capture on
```

F16 output：

```text
max absolute diff < 1e-7
```

---

# 28. Latent Extraction Schema

保存：

```text
latent_cache_v2/
seed_1/
    eth/
    hotel/
    univ/
    zara01/
    zara02/
```

每 sample：

```text
sample_id
L3 [20,D3]
L4 [20,D4]
L5 [20,D5]
native_logits [20]
```

不要保存 GT 到 latent cache。

GT labels 继续从原 feature bank 读取。

---

# 29. Latent Alignment Audit

新增：

```text
latent_alignment.json
```

检查：

```text
sample count
fold order
candidate order
native logit order
mode count = 20
sample_id alignment
```

要求全部 PASS。

---

# 30. P1-C：Global Latent Probe

第一轮不要上大模型。

Feature configs：

```text
G0
G0 + L3
G0 + L4
G0 + L5
G0 + G1 + L4
G0 + G1 + L5
G0 + G1 + G4 + L5
```

模型：

```text
Linear Residual
MLP Residual
```

---

# 31. Global Probe Protocol

严格：

```text
Outer 4 train / 1 test

Inner:
3 train / 1 val
```

调：

```text
alpha
```

不调大量网络结构。

---

# 32. Global Latent Probe 判定

当前 strongest global：

```text
0.5077m
```

所以：

```text
ADE >= 0.505
→ latent gain weak

ADE < 0.500
→ latent contains meaningful new selection information

ADE < 0.480
→ latent route strongly promising
```

---

# 33. P1-D：Local Latent Probe

只对：

```text
K=3
K=5
K=10
```

测试：

```text
G0 + GL + L4
G0 + GL + L5
G0 + G1 + GL + L5
```

模型：

```text
Linear Residual
MLP Residual
Tiny Set Attention
```

---

# 34. Local Latent 的重点不是全面 grid

只回答：

> Context-conditioned latent 能否提高 Top-K fine discrimination？

关键指标：

```text
Local Pairwise@5cm
Local Pairwise@10cm
Local Spearman
Selected ADE
Oracle Hit
NearOracle05
```

---

# 35. 最关键的 Local Discrimination 判定

当前 Native Top-5：

```text
Pairwise@5cm ≈ 64%
```

如果加入 L4/L5 后：

```text
>= 70%
```

说明 latent 有明显 fine-ranking signal。

如果：

```text
>= 75%
```

则非常值得继续。

如果仍：

```text
~64%~66%
```

说明 latent 也没有提供明显局部区分信息。

---

# 36. P1-E：Latent Ablation

如果 L5 有收益，做：

```text
L5 only
G0 + L5
G0 + G1 + L5
G0 + G4 + L5
G0 + G1 + G4 + L5
```

目的是判断：

```text
latent 本身
vs
trajectory geometry
vs
social risk
```

谁在提供增益。

---

# 37. 多 Seed 策略

当前仍先：

```text
Seed 1
```

只有当：

```text
ADE < 0.500
```

时，再跑：

```text
Seeds 1,2,3
```

避免在无信号方向浪费算力。

---

# 38. 新的 Final Decision Tree

## Case A

```text
L4/L5 Global ADE < 0.500
```

结论：

```text
F16 internal context-conditioned representation
contains deployable selection information
not captured by G0~G4
```

进入：

```text
Tiny Latent Selector
Distillation
Lightweight integration
```

---

## Case B

```text
Global >= 0.505
但
Local Pairwise 明显提升
```

例如：

```text
64% → 72%
```

结论：

```text
latent mainly helps fine discrimination
```

继续：

```text
Gate + Local Latent Reranker
```

---

## Case C

```text
Global >= 0.505
Local Pairwise <= 66%
```

说明：

```text
当前 F16 internal representation
也没有明显 recoverable selector signal
```

此时才进入：

```text
Large Capacity Probe
```

---

# 39. Large Probe 触发条件

只有满足：

```text
S6 post-hoc plateau
+
L4/L5 latent plateau
```

才做。

输入：

```text
history
neighbor history
20 candidates
native logits
L4/L5
predicted neighbor futures
```

禁止：

```text
future GT
ADE label at test
oracle index
```

---

# 40. README 当前应同步修改的结论

建议 Executive Summary 暂时改为：

> Validity-audited experiments show that F16 candidates are globally rankable, but fine discrimination among high-confidence candidates remains difficult. Native Top-K subsets contain substantial oracle headroom; however, current post-hoc geometric, confidence, social-risk, and local-context features recover only a small fraction of this headroom. Local reranking is competitive with strong global reranking but does not consistently surpass the strongest global residual baseline. The next phase therefore investigates whether context-conditioned decoder representations retain additional candidate-selection information that is absent from post-hoc trajectory features.

---

# 41. S6 Synthesis Report 建议的新最终结论

建议替换当前 Executive Scientific Verdict：

> The S6 series establishes a large gap between local candidate support and local candidate discriminability. Expanding Native Top-K substantially improves the local oracle bound, yet learned local rerankers recover only a small fraction of this potential. Neither hard-case weighting, near-oracle supervision, selective gating, nor Adaptive-K consistently moves the Macro-ADE frontier beyond the strongest global residual selector. This motivates a shift from post-hoc trajectory features toward context-conditioned internal decoder representations.

---

# 42. 暂停方向

完成本轮后暂停：

```text
更多 K sweep
更多 GL handcrafted feature
更多 hard-case weight
更多 epsilon/tau grid
更多 Adaptive-K threshold
更多普通 MLP depth/width
```

这些方向已经进入明显边际收益递减。

---

# 43. Agent 第一批执行任务

直接给 Agent：

```text
TASK 1
新增 reproducibility.py

TASK 2
统一 S6A/S6B/S6C training config

TASK 3
统一 neural training function

TASK 4
实现 S6R reproduction audit

TASK 5
修正 S6F/S6A/S6B/S6C/S6D/S6E 报告措辞

TASK 6
统一 collision_all / collision_social

TASK 7
实现 strongest-global-vs-local S6G baseline comparison

TASK 8
全部通过后，开始 S2-v2A L3 variance audit

TASK 9
生成 LATENT_HOOK_MAP.md

TASK 10
定位并提取 L4/L5

TASK 11
实现 Global Latent Probe

TASK 12
实现 Local Latent Probe

TASK 13
输出结果后暂停，不自动进入 Large Probe
```

---

# 44. 第一阶段最低交付物

```text
SELECTOR_LOCAL_RERANK_SERIES/
├── S6R_REPRO_AUDIT/results/
│   ├── reproduction_audit.csv
│   ├── reproduction_audit.json
│   └── RESULT_ANALYSIS_S6R.md
│
└── S6G_STRONG_BASELINE_COMPARISON/results/
    ├── strong_baseline_comparison.csv
    └── RESULT_ANALYSIS_S6G.md
```

---

# 45. 第二阶段最低交付物

```text
SELECTOR_RANKABILITY_SERIES/
└── S2V2_CONTEXT_LATENT/
    ├── LATENT_HOOK_MAP.md
    ├── results/
    │   ├── l3_variance_audit.json
    │   ├── latent_alignment.json
    │   ├── global_latent_probe.csv
    │   ├── local_latent_probe.csv
    │   ├── latent_ablation.csv
    │   └── RESULT_ANALYSIS_S2V2.md
```

---

# 46. 最终停止点

这轮 Agent 在生成：

```text
RESULT_ANALYSIS_S2V2.md
```

后必须暂停。

不要自动继续：

```text
Large Probe
Joint Training
重新训练 F16
```

先根据以下结果人工决定：

```text
Best Global Latent ADE
Best Local Latent ADE
Local Pairwise@5cm
Local Pairwise@10cm
L3/L4/L5 sample-conditioned statistics
```

---

# 47. 本轮最核心的科学问题

后续实验不再回答：

> 哪个 selector 架构更强？

而是回答：

> **F16 是否在内部已经编码了“某一候选为什么适合当前 sample”的上下文条件化语义，只是当前 confidence head 和 post-hoc selector 没有把这部分信息正确转化为 Top-1 ranking？**

这将决定 candidate selection 是否仍值得作为主研究方向继续推进。
