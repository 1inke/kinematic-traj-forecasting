# 代码补充核验记录

日期：2026-10-29。

状态：**合成CPU检查通过；真实数据／GPU／生产审核未执行。**

## 实际命令与结果

环境为项目现有 `.venv-governance/Scripts/python.exe`，未安装依赖。NumPy 1.25.2、SciPy 1.12.0、Torch 2.11.0+cu128。

```powershell
& '.venv-governance/Scripts/python.exe' -B 'docs/ENDPOINT_TEMPORAL_OBSERVABILITY_DESIGN_20261028/code/test_core.py'
```

最终运行：22 tests，5.238s，OK，退出0。此前增补期间18项、20项和21项版本也分别通过；以最终22项为本轮测试覆盖，不把多次运行视为独立实验重复。

另以 `ast.parse` 读取18个源码，全部语法通过；`run_stage.py`、`compare_models.py`、`combine_saved.py`、`freeze_sources.py` 的 `--help` 均退出0。PLAN_ONLY用不存在的输入路径验证，未读取这些输入、未创建指定输出目录。

## 测试覆盖

1. END49、TIME441数量／边界／identity。
2. 全部几何合法动作与独立标量重构比较。
3. 静止／零段长的TIME只identity。
4. 非法动作索引／NaN拒绝。
5. identity→参数范数→原索引的精确并列顺序。
6. 原生零阈值literal安全检查。
7. 运动量与标量实现比较。
8. safe教师包含identity、同槽J不劣于identity。
9. minADE/minFDE不同槽时不误报完整好轨迹。
10. raw／guarded保存路径与独立审核。
11. 修改连续参数、拒绝位或写入NaN会被审核拒绝。
12. P3双identity精确返回Native。
13. ID重复／GT乱序／track交叉拒绝。
14. train标准化与G标准化后屏蔽。
15. Native质量、聚簇及20条复制只计1个有效簇。
16. 整track/bootstrap计算对照直接展开窗口；空组不补0。
17. 人口误标、合成预检冒正式预检、缺少预检项／错误inventory被拒绝。
18. 终态收据缺失／失败阶段不得复用，正确终态协议可通过模拟接口。
19. 模型实际4770参数与零初始化。
20. 合成7窗的CPU中断序列化恢复EXACT，并拒绝换输入／换合同。
21. G屏蔽历史首层梯度为0、非屏蔽列可激活。
22. G/H合成8窗的独立NumPy全部梯度与固定两eps double差分。

第17、18项使用mock隔离门控，不冒称真实磁盘损坏恢复测试或正式source-lock验证。第20项仅合成7窗，不冒称实际513/12子集或全3103窗恢复。第22项不是实际GPU或量化／guard反传检查。

## 本轮工程修正

- CPU训练检查不主动获取CUDA RNG状态；真实CUDA训练仍保存／恢复CUDA RNG。
- 训练和指标读取教师文件时只读取需要的键，避免把全部动作J矩阵重复常驻内存。
- 保存审核增加连续参数→最近合法动作、全7失败位、shape／finite／整数ID检查。
- checkpoint补显式split／准备阶段绑定和训练合同。
- `done.json`不再单独代表可复用终态；源／资源检查之后才写 `STAGE_COMPLETE.json`，失败或半成品不冒完成。

## 没有发生的事情

没有新真实教师枚举、正式训练、实际GPU预检、全4411来源链重新核验、全人口独立完成审核、OS worker／自动任务启动、旧模型重训或文件删除。新脚本没有被包装成已经有科学结论的执行报告。

本记录是同助手的软件检查说明，不是外部评审，也不是项目性能提升证据。生产未完成项详见 [04_CODE_GUIDE.md](04_CODE_GUIDE.md#8-正式运行前仍缺什么)。
