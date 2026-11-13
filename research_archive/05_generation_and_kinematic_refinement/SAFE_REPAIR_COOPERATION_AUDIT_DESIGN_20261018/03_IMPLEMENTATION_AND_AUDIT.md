# 03｜实现与审核合同

状态：设计文档，以下伪代码未作为实验执行。只读CPU保存输出审计；不调用Torch、模型前向、求解器或训练。

## 1. 数据合同

沿W0 `JOBS.json`每条真实files路径读取，不假设父复用文件都已复制。640状态条目中只有18份actual路径进入消融。每份绑定native、gt、slot、sample_id、primal、raw、controls、correction、paths和原判定。

完整FIT `FULL_FIT_AUDIT.json/chosen`的key是数组index；必须通过portfolio的sample_id映射，不能拿index当真实ID。保留13个既定选择，不重新选择margin或消融表现更好的一份。

旧actual controls是生产重构的数值来源；独立递推可以检验controls→correction→paths，但不能把CPU重新tanh(raw)的末位差当成原GPU生产结果。保存源值，浮点重构容差继承原2e-12，不替代literal五motion门。

## 2. 核心伪代码

```python
def configurations(native, actual, target):
    # 所有数组float64副本；返回物理slot掩码与完整20×12×2路径。
    n, j = native.copy(), actual.copy()
    t = n.copy(); t[target] = j[target]
    o = j.copy(); o[target] = n[target]
    yield 'N', n
    yield 'J', j
    yield 'T', t
    yield 'O', o
    for k in range(20):
        if k != target:
            drop = j.copy(); drop[k] = n[k]
            add = t.copy(); add[k] = j[k]
            assert np.array_equal(drop[target], j[target])
            assert np.array_equal(add[target], j[target])
            yield f'DROP_{k}', drop
            yield f'ADD_{k}', add

def strict_gate(candidate_stats, native_stats):
    # 不计算带floor的ratio，不使用round()或评价epsilon。
    return all(candidate_stats[k] <= 1.01 * native_stats[k]
               for k in FIVE_MOTION_KEYS)
```

每个配置核验固定slot、目标路径不变、构造mask与保存controls一致、有限值、质量同slot以及位置界。所有42种配置保存，不因重复张量或失败丢弃行。

共18个case job及4个完整FIT portfolio job=22个分析job；756是配置记录，288是损失标量记录，均不是独立样本或训练次数。

## 3. 完整人口不能用chunk分位数平均

完整FIT按有界chunk读取/质量聚簇计算，运动学p95最终使用完整组元素的精确分位数。可用新目录内memmap保留有界数组，禁止chunk p95平均近似。四portfolio逐个处理，不同时缓存四个大张量。

样本质量、簇归属及有效good用原完整K20 SciPy complete-link规则；全FIT所有组均计算，不只检查13个阳性窗口。另存13窗口合并指标，呈现人口稀释作用。

## 4. 必须实际通过的预检

- native identity：全部slot不改、loss差0、所有预算0。
- 人工均值补偿：目标超预算、其他候选抵消；两种预算重构一致。
- p95非可加fixture：独立排序插值与原NumPy quantile一致，明确平均slot分位数会出错。
- j、a0、Qwinner、Jwinner可不同；相同目标路径的质量不随其他slot变化。
- 同窗口多个方案不重复算独立窗口；22057/slot7负例保留。
- 0维metadata不错误slice；原文件缺字段/ID错序/源SHA变更必须拒绝。
- 完成job仅当前source与全部输出SHA兼容才能skip；坏source和坏output注入测试应拒绝。
- 实际case按登记顺序取最前两份加既定负例做保存读回预检，不按效果挑例。
- 两个独立CPU实现复核质量、运动学、预算、objective、配置构造和全组汇总；不能只调用同一函数两遍冒充独立重算。

如果两个实现科学门不同，标记UNRESOLVED_NUMERIC并保存来源；不得加epsilon使门一致。常规读取/数值定位由执行助手处理，不把可自行解决的容差问题泛问用户。

## 5. 源锁、资源、ETA

继承W0原7580 source digests及1287输出证据；关联P1实际objective源码/协议和本轮新源。保留原digest，不覆盖旧hash；区分本轮关联范围与全部历史文件。runtime等可变控制文件不伪装成不变科学输入。

不沿用已被用户撤销的30分钟禁止、2秒求解截断或固定6GiB规则。当前任务与保留进程树按真实估计E、1.3E和动态系统余量管理；先测内存与磁盘，不OOM、不调OS/pagefile、不关闭用户程序。这里只是CPU审计，不需要安装SOCP依赖；这不是未来安装禁令。

实际授权实施时，以固定小批完整保存/聚簇/算术加源SHA吞吐估算22job与独立复核ETA，不能只按756条乘单次指标耗时。需长时后台时，沿既有唯一worker、一次startup、一次预计半程→完成节点流程；本设计交付不创建任务或查询任何旧PID。

## 6. 产物与完成定义

新执行子目录保存`protocol.json`、输入manifest、PREFLIGHT、case文件、configuration/objective表、四FIT汇总、ARITHMETIC_AUDIT、PRESERVATION、RESOURCE、RESULT及分析MD/JSON。

只有22job完整、756/288记录完整、原18方案判定重现、源和旧输出SHA前后不变、独立复核通过，才能ANALYZED。完成记录明确无新训练、无solver、无模型重放、非外部评审。协议锁后修改必须保留旧失败版本和新来源，不静默换hash。
