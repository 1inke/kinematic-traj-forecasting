# 解码端三方向代码包（2026-10-23）

此为2026-10-24新执行版本，人类最新“自行跑这三个方向的实验，有问题自己解决”覆盖此前只准备代码的状态。实际执行授权与版本修复见`EXECUTION_AUTHORIZATION.md`；以下10-23准备说明保留为历史背景，以`series_spec.json`、真实worker及新执行状态为准。原10-23代码包、原设计和旧科学结果均不修改。

## 三个独立目录

| 目录 | 入口 | 唯一机制 |
|---|---|---|
| `direction_A_time_basis` | `screen.py`、`train.py` | 先 knot3/FIT-svd4 零训练筛查，再条件 CV-free/CV-basis 配对训练 |
| `direction_B_scene_prior` | `prepare_records.py`、`probe.py` | 257桶、target-purge、严格时间资格的8逻辑臂历史终点探针 |
| `direction_C_set_mechanisms` | `run_c1.py`、`run_c2.py` | C1 stopgrad-aWTA 与 C2 SetMix 同形对照，分别实施、绝不叠加 |

共享 `common.py`（FIT/inner读入与同路径质量、全K20聚簇/五motion、旧A固定四子集）、`models.py`（精确网络/损失）、`training.py`（配对初值/顺序、全RNG断点、保存二实现审核）、`decisions.py`（原门）、`profile_decoder.py`（真实decoder测量）、`test_kernels.py`（合成测试）。

## 当前准备状态

- 代码已建立并执行合成CPU核函数测试及命令行入口检查。合成优化步仅用于测试零初始化梯度，不是实际FIT实验。
- **未**执行 A-Screen、B-Probe、任何实际FIT前向/反向/训练、实际GPU预检、完整旧源SHA复核、环境安装、OS任务或自动跟进。
- 不把代码单元测试写成 `PREFLIGHT PASS`。正式准备时才继承T1的615旧digest并扩入当前代码/合同；任一旧SHA变化即停止，不覆盖旧hash。
- 所有实际输出由调用者指定到该方向下的**新 execution 目录**。绝不覆盖旧完成执行。训练前必须另有明确、单实验包的人类授权记录，并通过当前真实预检；导入或 `--help` 不会训练。
- C2预检若没有可信 `FULL_PIPELINE_PROFILE.json`（冻结encoder+decoder的整模MAC/实际batch1、256延迟）则只记 `CORE_PASS_PENDING_FULL_PIPELINE_PROFILE`，不放行训练。decoder Linear-only MAC绝不冒充整模MAC。

## 常用检查（PowerShell，项目根目录）

```powershell
& .venv-governance/Scripts/python.exe -B qoder_work/DECODER_THREE_DIRECTION_20261023/verify_code.py
& .venv-governance/Scripts/python.exe -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_A_time_basis/screen.py --help
& .venv-governance/Scripts/python.exe -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_B_scene_prior/probe.py --help
& .venv-governance/Scripts/python.exe -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_C_set_mechanisms/run_c1.py --help
```

后续调用规则见各目录 README；本文不提供会自动串行跑全部方向的总启动脚本。

## 实施解释锁定（不改原设计文档）

1. SVD端点符号为0无法按原式定向时直接阻塞，不另找符号规则；两基底筛查并列时按原候选顺序优先knot3。
2. B的Python `hash()`跨进程随机，因此采用SHA256规范JSON `[sample_id,20261023]` 首8字节小端种子，排序后做置换；静态置换为一次FIT映射，不逐查询偷换种子。
3. B的Causal库限定**FIT**中已完结同序列片段，不向库灌入inner真实未来。未知时间保留ID并单列；若主497窗有未知查询，causal可核验子集仅描述，不能暗改完整主人口后报通过。
4. 元数据start_frame实为既有SOURCE索引。`t0=start+7/end=start+19`仅在已验证连续20索引记录使用，绝不声称未经核验的视频帧时间。原raw观测世界坐标必须显式提供与锁SHA，不用局部past猜世界cell。
5. 主体代码不会生成伪造的完整MAC、预检、授权或审查PASS。C2整模测量和B真实元数据核验均是运行前真实工作，不由合成测试代替。
6. 资源沿用任务估算E的1.3动态余量与Windows保留峰规则，无机械固定6GiB或30min门；E由新运行inventory明列后传入，不能静默增大以绕失败。

## 研究解释与审查

ARS研究设计核对用于保持单因素隔离、条件门、阴性分支及证据层级；本轮是同助手代码/合成测试审查，不是外审。旧F16已见inner，反复开发集、相关窗口和3seed限制全部保留。L2投影失败不是不可达证明，终点Opp4不是完整路径覆盖，同参数不是相同有效容量或优化几何；未通过完整G_base不晋升baseline。
