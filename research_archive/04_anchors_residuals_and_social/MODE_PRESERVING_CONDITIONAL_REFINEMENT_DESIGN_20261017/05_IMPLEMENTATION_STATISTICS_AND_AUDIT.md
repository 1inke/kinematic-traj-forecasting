# 05｜核心实现接口、统计与审核合同

状态：DESIGN_ONLY。以下为实现规范与伪代码，**不是已运行/通过的核心代码文件**；真正实施需在新execution子目录产生源锁及测试证据。

## 1. 模块边界

```text
execution/
  registry.py           # 旧immutable IDs/native/cache/source/stats逐项认证
  model.py              # 原120维25304参数共享head和U0映射
  objectives.py         # 唯一四臂分歧；禁止另加保护/物理/softmin项
  responsibility.py     # FIT-only Q0责任；并列规则与标签SHA
  train.py              # paired init/order、25epoch、完整RNG恢复
  diagnostics.py        # 固定256样本及batch责任计数，非训练数据筛选
  evaluate.py           # 原同路径quality/fullK聚簇/five motion
  bootstrap.py          # 固定families和同ID交叉seed×cluster bootstrap
  preflight.py          # 真实GPU/恢复/源/ID/保存路径测试
  audit_saved.py        # CPU第二实现，无训练或新模型前向
  worker.py             # 成功须包含最终保存审核与源保护
  start_independent.ps1 # 自有byte-lock、按需0trigger OS任务
  check_once.ps1        # 一次半程、一次完成，真实PID/parent/create/stage
```

W0另建solver client/service子目录，使用现有隔离求解环境。它和训练输出不相互提供teacher或超参数。旧模块只读引用/AST隔离，不能执行会在旧目录写文件的顶层脚本。

## 2. 核心前向与损失伪代码

```python
# 概念接口：shape B,K=20,T=12,XY=2；实际实现须逐位对照原U0。
def predict(observed96, native24, fit_stats, shared_head, native):
    x = concat(observed96.broadcast_to_slots(20), native24, axis=-1)
    raw32 = shared_head(normalize_like_locked_position_baseline(x, fit_stats))
    z = raw32.reshape(B, 20, 12, 2)                # 保持float32
    v = tanh(z).to(float64)                       # 原U0先float32 tanh，再double
    u = v / maximum(1.0, l2(v, axis=-1, keepdim=True))
    residual = einsum('tq,bkqd->bktd', locked_U0_kernel, u)
    return native.to(float64) + residual, u

def objective(pred, native, gt, fixed_native_owner, owner_rule, anchor_on):
    q = ade(pred, gt) + .5 * fde(pred, gt)          # [B,20]
    if owner_rule == 'dynamic':
        owner = first_physical_argmin(q.detach())
    else:
        owner = fixed_native_owner               # 训练标签，不进入predict
    task = gather(q, owner).mean()
    drift = ade_between(pred, native) + .5 * fde_between(pred, native)
    nonowner = ones_like(drift, bool)
    nonowner.scatter_(dim=1, index=owner[:, None], value=False)
    anchor = (drift * nonowner).sum() / (B * 19)
    diversity = mean190pairs(relu(.30 - meanpath_pair_distance(pred)) ** 2)
    loss = task + .05 * diversity + (1.0 if anchor_on else 0.0) * anchor
    return loss, task, diversity, anchor, owner
```

伪代码float64控制顺序必须以原U0生产表达式为准，不能单靠公式相似宣称bitwise。实际scalar与grad对照若发现不同，先定位dtype/表达式与设计含义，不暗改已锁文件。

补充要求：当前owner索引stop-gradient；固定owner由同一原生Q定义离线FIT生成。零初始化时A/B owner相同且anchor=0，四臂完整native20 identity。以后A/C owner可能切换；anchor不是detach(pred)，必须有梯度。推理函数禁止参数gt/group/scene/ID/logit/owner。

## 3. 实施前必须实际通过的检查

1. **ID与旧源**：各split唯一ID/native/GT/past/track/scene一致，无split交叉；P2按原时间区间资格；原模型与相关结果SHA继承，不重算覆盖旧digest。精确记录相关文件数，不能说核过全历史文件而实际只核依赖。
2. **配对/初始输出**：三seed四臂全部初始化权重/bool/stat/buffer EXACT；原生identity、完整20slots、位置界；A非零权重下前向/原L0 scalar与all-parameter gradients和既有U0实现EXACT。
3. **新目标**：NumPy独立scalar；mask分母B×19、first-slot ties、零漂移、非owner路径有梯度、owner不受anchor直接坐标梯度、shared参数仍可被其他路径改变；GT改变只影响loss，不改变predict。
4. **GPU实际连接**：固定前8个FIT batch、B=256，zero-last分阶段验证合法层正有限梯度；不以softmax解析抵消的roundoff冒有效梯度；本head无新增attention。
5. **有限差分**：固定seed非零double参数副本、eps1e-4/1e-5，两尺度误差relative≤1e-4或absolute≤1e-7；保存owner/单位盘活动面/tie。面切换样本标NONDIFFERENTIABLE，不比较不适用的光滑FD；固定32个hash参数方向，至少24个两尺度都无面切换且通过才足够。不得筛掉平滑失败、临时缩eps追PASS；不足则写证据不足并改新预注册版本，不假装原测试通过。不是float32量化证明。
6. **恢复**：四臂各真实513FIT+12inner子集两epoch，continuous/interrupted params全部buffers/optimizer/TorchCPUCUDA/loader/Python/NumPy RNG/history/新增诊断 EXACT；真实done SHA-skip与坏源、坏output、错owner标签拒绝。不是完整FIT两轮，也不是saved分析恢复。
7. **实际保存pipeline**：四臂各128固定outer样本，完整同路径quality、原SciPy聚簇、全部five motion、controls重构、source/outputSHA，CPU第二实现核对；smoke不计正式48jobs。
8. **资源/时间**：每臂4warm+32measure throwaway FIT256，加四population实际保存/聚簇/CI/终SHA成本。预估全部训练和审核，不只GPU吞吐；无benchmark模型保存或按效果选择参数。

实际脚本须固定函数与数组比较容差清单、随机顺序、环境版本并纳入source锁。科学good/物理门始终literal；数值算术复核容差不能混入评价。

## 4. 必须保存的数据

每run至少：`checkpoint.pt`、`run_manifest.json`、`split_ids`、`statistics`、`metrics.json`、`assignment_stats.json`、25行history及step diagnostics、done输出SHA。

Manifest包含：source与环境SHA、split/GT/cache/统计/owner标签SHA、seed、paired-init SHA、四臂规则、完整objective/分母/并列规则、optimizer与RNG、batchorder、epoch、dtype、U0核、全部outputs及hash。checkpoint绑定同一contract digest，不能只有裸参数。

每population保存：显式IDs、groups、tracks、scene/timeblock、native/GT/pred全路径、u及raw或可追溯control、逐slotADE/FDE/Q/J、原新good mask/owner/drift、聚簇标签、motion原始分量和汇总、rescue/lost列表。大数组按population/模型逐一处理，不把48个输出全部载入内存。

W0除逐题完整保存外，还保存实际生产重构证书和窗口/全FIT两个层级的物理检查。候选GT效用/owner只作训练或事后诊断，不能进部署接口。

## 5. 统计的精确定义

### P1 主family24

固定六对比 B−A、C−A、D−A、D−C、D−B、D−C−B+A；每个对比四终点：coverage率、lost率、rescue率、mean minJ。只对primary outer change作这一family；方向分别越大/越小/越大/越小越好。

100,000次PCG64(seed20261017)配对交叉bootstrap：每draw从3模型seed有放回抽3个；再从outer change的唯一target tracks有放回抽原数量个track，整track所有窗口保留；同一draw所有臂共用seed及track权重。每draw先按窗口权重计算每seed指标，再对抽得seed均值。不能把重复seed的预测当独立窗口。

每终点双侧Bonferroni percentile区间，端点α/(2×24)、1−α/(2×24)，α=.05，即约99.7917%单区间。完整draw数组保存，NumPy quantile method='linear'。稀疏conditional分母为零则该draw无效，不补抽；每终点至少95,000有效draw才可用于门，不缩小family；报告无效原因及数量。普通95%区间仅描述，不能替换校正门。

这是假设cluster bootstrap适用时的探索性多重比较控制，不是有限样本精确覆盖保证；极端分位数、仅3seed、窗口/scene相关性均限制可信度。

### 相对native与原成功门

B/C/D各自outer change coverage差共family3；B/C/D各自minADE、minFDE、minJ差共family9。使用同一paired draws，native在seed维重复只是共同参考，不是3个新native模型。固定α/(2m)区间，不按最终哪臂好删其他臂缩family。A−native、其他population及groups同时完整报告，标描述性。

P1→P2 gate仅看预指定D，不在B/C/D中挑最佳。每seed原重大百分比/5pp、protected/lost/motion门还要逐项检查；CI通过不能替代这些硬门。

### P2

主D−A四终点family4，primary cluster改为sequence×timeblock，seed与其交叉重采样，其余算法不变；target-track同算法是另列敏感性。D−native coverage/error仍保留上述family3/family9阈值，**不因仅D复核而变宽**。不同stage不合并窗口、seed或CI。

### 机制量

责任切换、漂移、distinct、effective-good、梯度cos等只描述，不新增无限p值或以它们绕过主coverage/安全门。若需要正式新机制检验，应另立协议，不能结果出来后改主假设。

## 6. 执行流程与资源

执行获准且全预检后，源码锁定，独立OS按需0trigger任务+自有byte-lock；一个真实startup核验PID/parent/creation/fullcommandline/task/stage/stderr，不用launcher壳PID充真实python。

每阶段实际预检后报预计总耗时及范围；仅一次50%预计elapsed检查，必要时据真实剩余训练/保存/审核修正一次完成节点。完成节点未完，按实际stage给新定点，不高频轮询、不每天重复旧节点。worker+exit双SUCCESS且全部审核齐全才ANALYZED。

实测任务需求E在启动前锁定；当前与Windows retained peak任务树private/RSS分别记录，按既有任务1.3E守卫。solver子进程不能只统计venv启动壳；动态系统physical/commit余量与剩余需求及并发实况记录，不回退机械6GiB规定。超过预算保留证据，不能运行后偷偷inflate E；正常实现修复在新目录复核，不改旧科学/失败记录。无OS/pagefile修改、无关闭用户程序。

本轮时间资源授权已取消30min/2sec硬否决；实现阶段不要再次要求用户批准普通依赖准备或已明确范围内的实现修复。实际源/ID不可解释、科学协议变更或扩旧模块权限才是需停下来说明的实质问题。

## 7. 完成审核不是只看退出码

P1完整12×25history、48科学输出、模型和manifest SHA；P2完整6×25、24输出；W0完整640状态及对应证据。独立CPU复算全部saved同路径质量、完整聚簇、五motion、责任/漂移、控制映射、bootstrap、原source/output/model SHA初末。

保存算术第二实现不是所有模型/梯度再次重放，也不是外部评审。若训练成功但保存/报告失败，只允许SHA兼容的saved recovery，不重复训练。最终`RESULT_ANALYSIS.md/json`必须分别给execution/audit/mechanism/major/replication状态，保留全部失败及未触发原因。
