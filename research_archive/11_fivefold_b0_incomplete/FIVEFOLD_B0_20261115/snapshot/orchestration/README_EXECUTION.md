# 五折 B0 并行执行更新

用户于2026-11-15要求暂停耗时较长的串行实验，改用 Downloads/1/FIVEFOLD_B0_20261115 (1) 中的新方案，并明确选择“准备并验证后直接启动”。design_original 保留新方案9文件。新代码只改 run_fivefold.py 的 ENVIRONMENT 多进程处理及并行说明，和对应 fake 并行测试；模型/训练/轮数/指标等均未改变。

## 改动与复用

独立控制器一次最多运行两个独立折/seed进程，共享同一张GPU。各进程保持 batch64、单线程、原模型、损失、AdamW、scheduler、EMA、增强与全部随机状态。10个预算run全部完成并封存决定后才并行15个最终run，再自动汇总。每个fold/seed只有一个所有者，不允许并行执行多个all调度器。

原ETH E64 seed42已经完成，核验DONE和所有文件SHA后复用；原ETH E128 seed42从完整epoch42/128、18900updates继续。其余尚未运行的折不凭空复用。科学输出仍保存到 FIVEFOLD_B0_EXEC_20261115/FIVEFOLD_RUNS，不复制或覆写已有科学结果。

为保证断点合同逐项一致，parallel_child 调用原封存的 run_fivefold/runtime/Trainer.step；新设计的并行建议由外层调度实现。旧源码和SOURCE_MANIFEST不改。正式ENVIRONMENT已经存在且来源已核验，后续进程只读，避开新代码中首个writer写入尚未完成时的读取竞态。新并行来源由本目录独立SOURCE_MANIFEST及逐作业receipt绑定，不伪装为改变模型训练代码。

旧源manifest233文件与新控制器独立封存。预算规则、64/128预算、三seed、五折、固定finalEMA、完整float32预测/逐窗指标保持。用户之前批准的录制隔离A3修订继续有效，原始失败F0不改。F0完整性审核读取测试数组，不用于模型预算选择。EXP156/133仍只是不同协议量级对照。本次不自动上传GitHub。

## 已做的真实预检

串行两个128update任务合计31.93秒，两个并行16.32秒，短测加速1.96倍；所有最终model/EMA/optimizer/scheduler/RNG/order/history逐项EXACT。该数字是短测，不是整项研究的保证倍率。两进程实际CUDA张量各峰约194.5MB、缓存247.5MB，不包含CUDA上下文/其他应用显存；内存和commit另按原warning-first策略监测，不把模型显存当进程总占用。

并行status写入各自slots，父进程聚合CURRENT，避免共享.tmp竞争。使用Windows Job Object的KILL_ON_JOB_CLOSE，子进程在挂起时先加入job再恢复，包含venv转发器的全部后代；强制终止owner的真实子进程退出测试通过后才启动。失败时只停止自有子进程，不盲重试。

## 启动、状态、暂停

本目录standalone/START.cmd为新的启动/恢复入口，STATUS.cmd只读看多任务进度，PAUSE.cmd停止全部自有进程并校验已提交断点。旧串行Windows任务停用，防止旧入口重开副本；新旧控制器共用同一命名mutex。

新按需任务Trajectory-Fivefold-B0-Parallel-20261115零定时trigger、无限运行时限、IgnoreNew，不依赖Codex/GPT或网络。可以退出Codex，需电脑开机保持登录。运行期仅临时阻止系统自动睡眠，屏幕可关闭；关机/注销/手动睡眠会中断，开机登录后手动START恢复最近完整epoch，中断当轮重算。保留每run最近2pt及SHA证据；不删旧实验。

最新状态优先读本目录standalone/CURRENT.json、CHILD_STATE.json、WORKER_EXIT.json，再核实际task及进程。原执行根standalone会镜像新状态，但旧pause_events仍是历史。成功须25份DONE及输出SHA、SUMMARY、FINAL_OUTPUT_SEAL、RESULT SUCCESS和WORKER_EXIT exit0同时满足。
