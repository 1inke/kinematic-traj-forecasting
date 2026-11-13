# DGeo 几何感知模式协同：有限四臂执行版

实际状态以 runtime_control/WORKER_STATE/WORKER_EXIT/RESULT 与 GEO_EXTRA_REVIEW 为准，不因 README 快照重复 worker。用户授权：根据两份方向文档执行，冲突沿已确认协议，不再因旧维度/梯度描述机械阻塞。

## 四臂

|arm|相对比较|总参数|
|---|---|---:|
|DG-Base|新工程复合对照：目标槽16D协调/非零gain.1出口/原mode打分输入|301922|
|DG-GeoGate|对Base加12D几何及QK跨模式中心化|302754|
|DG-GeoDiff|对GeoGate仅换A1-.5A2算子|304802|
|DG-GeoScore|对GeoGate仅加后验关系logit残差|304172|

每臂whole-model fresh seed42固定8轮2720步，共32轮10880科学更新。原C0 fit21719/dev4753、B064D/clip5、loss/AdamW/EMA/scheduler/增强/原params初始化/全部批序和每epoch所有RNG不变；B0只读参考，不加载其训练权重warmstart，不新U或组合，不访问test。

实现使用实际h3接口：coord只目标槽，邻居仍参与F16编码；12D几何只由当前尺度锚点和过去坐标计算，GT不进预测。phi中速度量按方向公式为每sample位移（非m/s），原运动评价仍float64 dt.4。原score mode与回归mode部分解耦，但relative锚点仍共享，不能宣完全梯度解耦。

## 合同优先级和 ARS DA1

见 AUTHORIZATION_AND_ALIGNMENT.md。用户原确认设置优先，新文档数字为待测陈述；不强迫复现-.0466余弦/41.88%负权或特定梯度范数，不调真实样本/epsilon/初始化求门PASS。实测四臂初始B0paths/logits EXACT、每个新增张量第2步梯度非零；保留完整前3步，沿旧实际分阶段连通门，不把首步0改写成永久死锁。

原tail零参考保持NOT_ESTIMABLE_NO_FLOOR，新增可估单元单列、全组ADE保护单列，不改旧主门。全候选complete-link+pdist/同slotADE,FDE,J/五pooled+junction/原成功丢失和逐ID裕量全部保存。关系打分另报两模型共同覆盖及三模型交集分母，不拿集合变动夸排序。

ARS同助手DA1/2/3非外审；报告中不同Top1/分组数据以本地原数组为准；“未过观测机动阈值”不是信息论不可预测，“负权少”不是秩1或去噪失败的因果证明。额外容量/优化几何/部分评分解耦均为竞争解释。单seed/同track相关窗/反复dev/姐妹footage未知限制保留，不宣显著性或SOTA。

## 已执行实际预检

四arm GPU8+生产first3batch、paired原权重/buffers/RNG/native初值、独立NumPy/functional全新增参数梯度及固定2方向×两eps FD、513真实FIT两轮18更新/11中断所有params/EMA/Adam/scheduler/CPUCUDA-Python-NumPy-loaderRNG/order/history精确恢复；128保存完整质量/原聚簇/运动双实现、坏source/split/stat/ID/output读取拒绝。另各340步真实完整FIT第一轮对B0全部RNG/order EXACT；fixed128初始机制、晚加入共同覆盖/无条件quantile消费者实际运行。前检256+额外回归1360更新是夹具，不是四科学模型训练。

完整成本各arm raw过去因果邻居→transfer→heading→F16+h3+coord/scorer→inverse→top1 CPU1thread/GPU1/64各50warm500，包含attention contraction的ATen FLOP与所有operators，不冒Linear-only为整模；非线性/NumPy scalar FLOP不计但延迟含，非部署证书。

本轮方向文档明确45min全流程排期，按本轮实测+上一UD真实生产计时保守校准；旧UD36.46min预算偏差不重写，原科学验收门不变。资源仍warning-first v3，1.3仅预警，实测峰+inventory给E，不fixed6GiB/锁内存/改OS/pagefile/kill用户程序。零删除，所有旧PAUSED保持。

## 独立执行与完整完成

唯一XML0按需Windows任务串行四个train_arm.py→compare_geo.py→geo_extra_review.py；每epochcheckpoint及全部RNG可恢复，但不自动盲重试。完整完成必须worker/exit双SUCCESS、4final/32epoch10880、8科学人口所有done输出SHA和独立保存审核、源终SHA、RESOURCE、共同支持与全部无条件尾部的GEO_EXTRA_REVIEW、FINAL_SEAL。仅RESULT出现但worker仍额外审核不算完成。

生成RESULT_ANALYSIS.md/json以及DGEO_RESULT_ANALYSIS.md、ALL_GROUPS、PAIRED_COMPARISON_VS_B0/controls、PILOT_SCREEN、DGEO_MECHANISM_DIAGNOSIS、COMMON_COVERED_SUPPORT_TOP1、UNCONDITIONAL_TAILS。三seed只标满足原门后可申请独立准入，后台不自动启动；不新loss/更多变体/旧解冻/data，baseline/historycycle11/长期goal不重置。
