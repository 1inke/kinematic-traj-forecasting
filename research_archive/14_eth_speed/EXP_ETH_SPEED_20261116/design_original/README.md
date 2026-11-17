# ETH 速度诊断与速度增强探索实验

日期：2026-11-16。交付内容：两个实验的设计、实现、测试和本机配置。**本轮不启动正式诊断、模型测试推理或真实数据训练。**

先阅读 [EXP_ETH_DIAG.md](EXP_ETH_DIAG.md)，再阅读条件实验 [EXP_SPEED_AUG.md](EXP_SPEED_AUG.md)。研究问题、固定条件和解释边界见这两份文件。已有 ETH-T1 实验完成；旧五折继续暂停；COMBO 继续搁置。

## 这次安排

| 实验 | 新训练 | 复用内容 | 主要问题 |
|---|---:|---|---|
| EXP-ETH-DIAG | 0 | ETH/HOTEL 的 B0 E64 seed42；ETH 已有 B0/T1 预测 | 速度差是否来自当前预处理；只改观测速度是否有帮助 |
| EXP-SPEED-AUG | 条件满足后 1 次 | ETH 同索引、同预算 B0 E64 seed42 | 将速度增强扩大到 0.5–2.0 是否改善 ETH |

执行默认单路。诊断完成只写训练建议，**不会自动开始增强训练**。没有自动补 seed、自动扩大训练轮数或自动恢复旧实验。

## 文件

- `code/run_eth_speed.py`：`prepare` / `diag` / `aug` 三个独立入口。
- `code/audit_eth_speed.py`：原始标签、五折数组、取帧脚本和视频元信息的审核。
- `code/eth_speed_math.py`：观测速度、训练分位数、缩放与逆变换、机动分组、固定筛选条件。
- `code/speed_aug_adapter.py`：只替换原 B0 的增强范围，复用原训练器及完整恢复状态。
- `code/report_eth_speed.py`：由诊断结果输出中文报告与分层 CSV。
- `code/validate_runtime.py`：实际 B0 引擎的合成数据预检，不读取基准轨迹。
- `tests/test_design.py`：变换、泄漏边界、原增强函数、诊断流程等 CPU 合成测试。
- `CONFIG.local.json`：已填写本机路径和已有模型/预测的 SHA256。
- `standalone/`：手动启动、暂停、状态脚本，通过 Windows 任务计划程序脱离 GPT 运行。
- `qa/`：本轮验证证据；不属于正式实验结果。

这是一套复用本机现有项目的实现，不含模型权重或原始数据。依赖当前 `.venv-governance` 环境中的 PyTorch、NumPy、OpenCV，以及配置所指的原 B0 / 五折代码。换电脑时须一并迁移这些依赖，修改配置并重新核验；不能只复制本目录就直接复现。`qa/create_local_config.py` 仅是工作区内的编写辅助工具，日常执行不需要运行。

## 执行方式

在此目录打开 PowerShell。下面使用本机已经安装的环境，无需重新安装软件包。

```powershell
$py = 'C:\Users\linke\Documents\New project\.venv-governance\Scripts\python.exe'
& $py -X utf8 -B tests/test_design.py
& $py -X utf8 -B code/validate_runtime.py --device cpu --out qa/CPU_RUNTIME_PREFLIGHT.json
# 正式使用 GPU 前，可先对目标设备做同样的合成数据预检：
& $py -X utf8 -B code/validate_runtime.py --device cuda:0 --out qa/CUDA_RUNTIME_PREFLIGHT.json
```

需要执行实验时：

1. 运行 `standalone/PREPARE.cmd` 可单独做原始数据审核；不加载模型。
2. 运行 `standalone/START_DIAG.cmd`：自动先核验依赖和数据链，再做零训练诊断。
3. 查看 `standalone/STATUS.cmd`，以及输出目录中的 `DIAG/RESULTS_ZH.md` 和 `DIAG/DECISION.json`。
4. 仅在诊断为 `ELIGIBLE_EXPLORATORY`、决定实际开展一次训练后，运行 `standalone/START_AUG.cmd`。不满足条件时程序拒绝训练。

也可前台运行，语义相同，但关闭终端会中断前台进程：

```powershell
& $py -X utf8 -B code/run_eth_speed.py prepare
& $py -X utf8 -B code/run_eth_speed.py diag
& $py -X utf8 -B code/run_eth_speed.py aug
```

不要一次粘贴以上三行来代替阅读诊断结果。`aug` 是单独的手动操作。

## 退出 GPT、暂停与恢复

后台任务名为 `Trajectory-ETH-Speed-20261116`；只有手动启动，没有定时触发或重启后自动训练。任务由 Windows 任务计划程序持有，退出 Codex 不应停止任务；电脑仍须开机、保持唤醒和登录。

运行 `standalone/PAUSE.cmd` 请求暂停。训练完成当前轮次并保存完整状态后退出；诊断在当前条件完成后退出。**等待 STATUS 显示 `PAUSED` 且任务已不处于 Running，再关电脑。** `LAST_EXIT.json` 可能来自上一次会话，要结合任务状态和最新时间查看。

重新运行相应 START 脚本会清除暂停标记，核对输入合同，从最近完整轮次或条件恢复。强制关机丢失当前未完成轮次或条件；不会把半个轮次算作完成。每次会话最多约 2.5 小时，在边界保存后退出。共享互斥锁防止和旧五折控制器同时执行；不结束其他程序，不修改旧实验。

## 输出与完整性

默认新输出目录：`C:/Users/linke/Documents/New project/qoder_work/EXP_ETH_SPEED_EXEC_20261116`。正式运行前尚不存在是正常状态。

`INPUT_CONTRACT.json` 固定设计源码、本机配置、旧封存依赖、五折数组、标签、视频、模型和预测的散列。恢复时输入有变化即停止，需使用新的输出目录，不能混用结果。工作区版和下载版的源文件路径不同；选择其中一份作为执行入口后保持一致。

诊断输出包括预处理审核、训练速度统计、每窗缩放比例、原世界坐标预测、逐窗指标、按速度/机动/原分组/速度×机动汇总、中文报告和建议。分层 CSV 为 UTF-8 BOM，可直接导入表格软件。指标从保存的 float32 预测重算。

增强训练保留最近两个完整轮次的权重、优化器、EMA、调度器、损失缓冲、所有 RNG、批次顺序状态；最终保存验证集和测试集预测、分层指标、比较结论。完成以对应 `DONE.json` 输出散列核验通过为准；进程消失或训练进度到 100% 本身不代表评价完成。

## 时间与验证范围

零训练诊断暂按 **3–10 分钟**安排；包含源/数据核验、原始标签匹配和约 7805 窗次模型推理，实际未测速。增强单次 64 轮暂按 **40–75 分钟，加少量评价时间**安排，参考此前 ETH B0 约 39.6 分钟、ETH-T1 约 68.3 分钟；不承诺固定结束时间。

CPU 测试与合成数据的实际 B0 训练器预检见 `qa`。尚未执行目标 GPU 的预检、正式原始数据审核、ETH/HOTEL 新测试推理或增强训练，也没有新科学结论。源码设计审查见 [DESIGN_REVIEW.md](DESIGN_REVIEW.md)。
