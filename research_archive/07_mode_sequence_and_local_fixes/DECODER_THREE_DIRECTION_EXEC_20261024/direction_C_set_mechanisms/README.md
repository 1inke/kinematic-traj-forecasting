# 方向C：C1与C2各自独立

状态：`CODE_PREPARED_NOT_LAUNCHED`。同目录是因为属于方向C，不表示两实验可叠加或连续自动开跑。

- `run_c1.py`：82008参数原ContextK20；hard-WTA vs detach-softmax aWTA，592step固定0.20→0.02米退火，endpoint190对**线性**hinge不变。保存软责任/熵/硬槽活跃/跨轮切换。hard控制实际输出必须旧A逐位复现，否则停止，不重训凑匹配。
- `run_c2.py`：98552参数同形两臂，原主干不缩减；384→32→128残差末层零，self/mean/amax vs self/self/self。整模365711，明确超305000预算。合成Jacobian、amax并列分梯度、置换等变和分阶段激活已纳入测试。

两入口均支持 `--prepare`、`--preflight`、`--train`，其中**一次只选C1或C2**；其余选项同方向A，不需要`--screen`。训练要求真实新指令授权JSON（package为C1或C2），匹配的实际GPU/2epoch全RNG/保存审核及源锁；当前没有这些生产PASS或授权，不启动。

C2必须补齐真实`FULL_PIPELINE_PROFILE.json`：每臂`whole_parameters=365711`、`full_model_MAC_per_window`、`includes_frozen_encoder=true`、`counts_all_executed_matmul_convolution_ops=true`，以及`latency`下`1/256`的实际GPU秒/批；源digest须匹配。这是测量产物合同，**不可手填估值冒充实测**。`profile_decoder.py`只可测student Linear下界/延迟/峰值，不会生成整模PASS。缺完整测量，预检明确`CORE_PASS_PENDING_FULL_PIPELINE_PROFILE`，训练被拒绝。

原始C1/C2门、每臂全部24组G_base、原生success lost、连续误差和全K20五motion独立报告；局部阳性不能豁免原accel_mean门、不自动接B/E/scorer或扫温度/层数。
