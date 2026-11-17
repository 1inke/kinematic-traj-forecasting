# 机动模板零训练检验：完成，NOT_SUPPORTED

**Execution COMPLETE / scientific verdict `NOT_SUPPORTED`。** 2026-11-17完成并归档。五折×10配置（四主臂、六敏感性）×三个聚类seed，共150次k-means，正式计算322.2秒；无网络训练、模型推理或GPU使用。

[结果解读（中文）](RESULT_INTERPRETATION_ZH.md) · [原完整报告](REPORT.md) · [全部数值](snapshot/RESULT.json) · [状态](STATUS.json) · [原设计](control/design_original/README.md) · [执行说明](control/EXECUTION_NOTES.md)

## 结论

K表示三段速度比和方向，P为绝对位置模板，P_AUG为增强后的位置模板，S为按当前有效速度归一化的位置模板。四臂同用4/8/12关键点与B0线性插值，K=20，聚类只使用各折95% fit。

| 固定检查 | 结果 | 数值依据 |
|---|---|---|
| C1 ETH收益 | 通过 | K相对P：ADE改善27.73%，FDE改善21.07% |
| C2 相对增强 | 未通过 | 五折test等权FDE：K .58994，P_AUG .55585 |
| C3 不伤其他折 | 未通过 | 其他四折FDE恶化2.94%–30.42% |
| C4 分布内 | 未通过 | 五折val等权ADE/FDE恶化13.30%/25.37% |
| C5 减速/停止 | 未通过 | 合并FDE：K .48642，P .43188，P_AUG .50463 |
| C6 三seed一致 | 通过 | ETH三个聚类seed均优于对应P |

因此不支持直接为本次固定20模板的K表示投入网络训练。这个结论不否定所有运动学解码。ETH单折改善不能抵消其他折、验证集与减速组的退步；S和P_AUG在ETH FDE也优于K。归因K相对S、S相对P均为MIXED。敏感性只报告，不替换主V0=.3或修改判定。

`REF_P_ORACLE_SCALE`是用真值在17个固定尺度中按ADE选比例的参考，minFDE取同一比例下的独立候选最小值；不是连续尺度的严格上界，也不能部署。已有训练B0/AUG仅作参考。三个seed是聚类初始化重复，不是三次神经网络训练。

## 归档与复核

- `control/`：原字节设计和5份代码、自检、输入校验、47文件源合同、后台入口、服务父链、执行日志、独立几何核验代码和回执。
- `snapshot/`：原14份科学输出，以及后续独立核验派生误差、解释和来源；原结果和阈值均保持。
- **21份NPZ**：150次聚类的模板数组（每次20中心）、10份原逐窗seed平均误差、10份逐seed误差。所有主臂、敏感性、尺度参考和已有模型参考的分组分析可复算，无原始数据或完整候选坐标轨迹。
- `dependencies/`、`COPY_PROVENANCE.json`、`OMITTED_ARTIFACTS.json`和`ARTIFACT_MANIFEST.json`保存实际依赖、副本映射、省略资产及公开文件散列。

模板本身是fit聚类中心，P/S/K参数有明确含义；它们不是原始单个窗口坐标。逐窗口文件中的标签、速度、误差是派生评价量。原始数据、完整世界坐标预测和权重继续留本机。

本机已独立从原始数据和150份模板重建全部候选、插值和误差，核对五折val/test全部分组、六敏感性、尺度参考及C1–C6、判定数字、归因，全部PASS；未重新聚类。原封存和发布前SHA校验记录保留。

公开CPU工具从逐seed派生误差重新汇总所有分组的ADE/FDE、p90和可达率、合并组、六项检查与判定，并检查原逐窗平均误差和模板说明。**公开复核属于派生误差复算，不能重做缺少原始坐标的几何核验、重新生成标签或重新拟合聚类。**

```bash
python -m pip install -r requirements-analysis.txt
python -B tools/reanalyze.py
python -B control/design_original/code/test_kine_oracle.py
```

Python 3.10+与NumPy即可，不需要GPU、PyTorch、原数据或网络。p90/可达率须逐seed计算后平均，不能从seed平均误差反推。工具不会启动正式实验。

原设计中的“待执行”和执行说明中的“尚不发布”是准备阶段记录；当前实验已完成，并按用户后续明确要求同步GitHub。保留的绝对路径、任务XML和原脚本只作执行来源，重新完整运行需要本机原数据、重定位配置和新的输出目录。

探索局限包括反复查看的测试数据、有限20模板、角度欧氏聚类及固定低速先验。没有新训练、自动组合、baseline升级或旧五折恢复。

前次 [减速诊断](../../15_decel_diag/EXP_DECEL_DIAG_20261117/) · [速度诊断与增强](../../14_eth_speed/EXP_ETH_SPEED_20261116/)
