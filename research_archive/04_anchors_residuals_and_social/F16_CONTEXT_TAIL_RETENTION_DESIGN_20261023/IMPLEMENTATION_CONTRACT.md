# 实现与验收合同（不是已实现代码）

本文件提供模块接口和核心伪代码；未提供可直接启动的launcher，未执行GPU、训练或科学输出计算。设计与protocol.draft.json冲突时先修正文档，不默选较宽松要求。

## 1. 输入与来源

项目根：`<WORKSPACE_ROOT>`。

旧执行根：`qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022`。

需要绑定：旧protocol/core/training/input_contract、完整source_manifest及其source extension、statistics.npz、split_ids.npz、三个initial_states和batch_orders、三run checkpoints/done/history、六predictions与sample_metrics、两native输出、最终报告及posthoc_interpretation。

执行前沿原manifest逐项验证并继承原digest；新增来源单独扩展，不能重新hash旧文件掩盖变动。只锁与此次研究相关源和数据，不宣称全部历史目录已核验。新实现禁止直接import具有旧目录写入副作用的legacy入口；只读AST提取或新shim均需实际等价验证。

最低数组合同：

```text
sample_id: [N]，unique，完整按ID连接，禁止仅假设当前行顺序相同
groups: [N]，0/1/2沿原定义
track: [N]，原sequence+target身份，不用局部数字跨scene碰撞
past: [N,8,2] float32，最后观测原点；若从cache取按显式ID验证
tokens: [N,8,96] float32，同实际F16 producer
native: [N,20,12,2]，immutable
gt: [N,12,2]，同坐标米
student_paths: [N,20,12,2] float32生产，评价转float64
```

所有非有限/重复/身份不明立刻保存错误，不删除难例继续。无邻居重新选择、原始label补全、新GT字段、外部数据。

## 2. 模块划分

```text
execution_<date>/
  source_contract.py      # manifest/IDs/统计/初始状态/顺序绑定
  saved_diagnostics.py    # D0/D1，只NumPy保存数据
  student.py              # 精确82008原模型，无旧目录写副作用
  losses.py               # base和relative_J，唯一训练差异
  preflight.py            # 实际梯度、resume、保存审核
  train_paired.py         # 只六个fresh runs；固定epoch8
  evaluate_saved.py       # 全K20质量/聚簇/运动/CI
  independent_audit.py    # 独立算法重算，不调用生产汇总函数作独立证据
  protocol.json / source_manifest.json / runtime_state.json
  D0/ D1/ runs/ completion_review/ failures/
```

科学源在预检后锁定。runtime、报告、控制记录与锁源分开。发现实现错误必须保存旧失败版本和差异，不能静默改hash；科学定义变化需新设计，不以“修复”名义改变协议。

## 3. 核心训练逻辑示意

以下只是实现约定，需实际Torch/NumPy/梯度验证，不能复制后即宣称正确。

```python
def quality(pred, gt):  # pred [B,20,12,2], gt [B,12,2]
    e = torch.linalg.vector_norm(pred - gt[:, None], dim=-1)
    ade, fde = e.mean(dim=-1), e[..., -1]
    q = ade + 0.5 * fde
    # component tie ADE first; candidate tie first original slot.
    j = torch.stack((ade / 0.2, fde / 0.4), dim=-1).max(dim=-1).values
    return ade, fde, q, j

def objective(pred, gt, native_j, arm, pair_i, pair_j):
    ade, fde, q, j = quality(pred, gt)
    qmin, qslot = q.min(dim=1)
    jmin, jslot = j.min(dim=1)
    endpoint = pred[:, :, -1]
    distances = torch.linalg.vector_norm(
        endpoint[:, pair_i] - endpoint[:, pair_j], dim=-1)
    diversity = torch.relu(0.3 - distances).mean()  # NOT squared
    base = qmin.mean() + 0.05 * diversity
    excess = torch.relu(jmin - native_j.detach())
    if arm == 'A_replay':
        total = base  # control scalar/gradient must match original exactly
    elif arm == 'P_relative':
        total = base + 0.2 * excess.mean()
    else:
        raise ValueError(arm)
    return total, {'Q_slot': qslot, 'J_slot': jslot,
                   'relative_excess': excess.detach()}
```

native_j为FIT原生路径和GT在**相同float32训练算术**下计算的min-slot J，detach且有限；预计算文件按ID绑定dtype/sourceSHA。float64评价重算单独保存，两种算术不能混称bitwise相同。生产与预检统一compute batch256，最后batch训练按原训练行为、不擅改drop_last；推理按原实现补零到256，仅真实行进入评价。

模型forward只收210维观测特征。`native_j`、native路径、GT、group、ID、teacher、scene、logits一概不输入模型。没有部署时native比较器、GT门控或挑选更优teacher。

初始化同旧模型是20条零轨迹，不是native20 identity；零头初步前层梯度可以为0，必须分阶段验证，不伪造全层首步正梯度。

## 4. D0/D1计算约定

```python
J_slot = J.argmin(axis=1)               # 一窗一个slot
chosen = paths[np.arange(N), J_slot]  # 全时间共同使用
success = J.min(axis=1) <= 1.0
retained = native_success & success
rescue = ~native_success & success
lost = native_success & ~success
neither = ~native_success & ~success

# 固定最后观测原点；原五motion为全K20，不是只对chosen
velocity = np.diff(np.concatenate([zero_origin, paths], axis=2), axis=2) / 0.4
acceleration = np.diff(velocity, axis=2) / 0.4
jerk = np.diff(acceleration, axis=2) / 0.4
```

all-K20规范：speed [N,20,12]，accel [N,20,11]，jerk [N,20,10]；每group直接pool全部元素再计算linear p95。不可把每window p95平均作为pooled p95，不可只算successful slots。

保留GT-free原SciPy complete-link，D=max(meanpath_distance/.10,endpoint_distance/.20)，原cut规则不变。effective_good需要同路径成功标签与GT-free簇共同计算，不以簇数越大越好。

## 5. 最低实际预检

1. 源/ID：继承manifest，三seed两split全部对应；统计只FIT。计算数据资格不由误差筛选。
2. CPU fixtures：同路径ADE/FDE不能跨slot拼合；四类partition加总；零速度/零位移角度NA；共同失败误差贡献正负均正确；时间运输量两零与单零分支；constant-speed/单拐点/反向/endpoint-only误差。
3. 损失：control对原A scalar和全部参数gradient等价；P的Torch/NumPy forward在double绝对差≤1e−10；scalar项按实际batchN聚合，不平均batch均值。
4. 固定非零fixture两eps 1e−4、1e−5参数方向FD，绝对误差≤1e−6+1e−4×max(|数值|,|解析|)。记录Q/J winner、J component、hinge active集合；跨面就标FD_NOT_APPLICABLE并停止该预检，不搜索eps/seed求PASS。另用解析分段fixtures检验ties和hinge0，不能拿不可导点FD当失败根因或伪造PASS。
5. 实际GPU8 FIT256分阶段合法梯度有限且连接，完整paired weights/buffers/statistics；原冻encoder若不实例化则只能声称cache/source保护，不声称旧GPU参数再次重放。
6. BOTH真实FIT513+inner12子集两epoch continuous/interrupted：全部参数buffers、AdamW状态、Torch CPU/CUDA/loader/NumPy/Python RNG、history/新增诊断EXACT。不是fullFIT两轮，也不是分析恢复。
7. 每臂实际固定hash128保存路径、全质量/完整原聚簇/五motion独立算术、ID和误差分解；float64同路径量建议绝对容差1e−10，整型/bool/索引EXACT。数值容差仅audit比较，评价成功线和1%门不加epsilon。
8. 完成run source/output SHA兼容才能skip；坏source/output/统计/初值/顺序实际拒绝。旧A不是completed-P，不能用错arm的checkpoint跳过。
9. 诊断不得推进训练RNG；仅需要loss的GT，运行forward-signature和输入依赖测试确认部署无GT/native_j。
10. 实际资源/吞吐测量、旧源数据SHA初末不变；所有门PASS才能锁定并启动。预检失败保留全部来源，不自动科学变更。

## 6. 必存产物

每run：checkpoint.pt、run_manifest.json、history.json、assignment_stats.json、split_ids.npz、statistics.npz、两split预测/逐窗指标/doneSHA。

run_manifest至少包含source完整digest/实际源清单、split IDs、统计、初始权重、batch order、seed、模型参数量、base objective公式、aux公式/权重/单位、dtype、训练epoch/steps、optimizer全配置、clip5、环境版本、原生参考J来源和计算dtype、是否有teacher坐标loss（本轮false）。

history：base/task/div/relative penalty、hinge-active比例与均值、Q/J责任直方图及切换比例、extra tie数量、component tie、preclip norm与clip比例。A也记录reference excess但不能改变优化。原native成功/失败FIT分层loss只作记录，不分组加权。

D0：逐窗ID/四类标签/各slot质量/Jslot/Qslot/端点方向幅度/早中晚/shape error/贡献/尾部描述。

D1：all-K20 pooled五motion、chosen与GT的v/a/j向量误差、11项acc时刻量、时间质量W与NA、观测-预测衔接、各组/seed分母。

CI：全部draw定义、track/seed索引或可重建随机状态、无效draw数、所有16次级结果；不能只留显著项。

R：RESULT_ANALYSIS.md/json、DECISION.json、资源峰值/实际耗时、全部source/output SHA、CPU独立审核。列出未执行B/E、未做独立wholemodel测试、未完成全端到端MAC；不给“整套创新已完成”状态。

## 7. 独立审核与CPU成本

D0/D1共享计算、一次生产和一次独立复算；复用完整旧聚簇结果需核对输出SHA并明确不是再重聚类。新T1每份预测需完成原全质量/聚簇/运动审核。独立实现不得只调用同生产summary函数后宣称独立。

chunk256/有界内存，一次一个人口/模型；先按track聚合bootstrap充分统计，避免每draw复制全部路径。bootstrap mean/coverage/lost可用sum/count统计，motion p95不在此CI集合，不得用均值统计伪造分位数。

actual ETA包括读取、保存、全部聚簇、CI、SHA和审核；任务E须按数组、checkpoint、模型、optimizer、临时IO估计。执行不得借设计文档跳过资源guard或把预检峰值充当生产峰值。

## 8. 可复现验收的限制

保存算术不是所有模型和梯度二次重放；source当前一致不能补成过去未记录的训练锁源。旧A fresh重复也不是独立新泛化。未经完整stage门不能启动B/E_bal、scorer、原模型解冻、外部数据或第四局部变体。
