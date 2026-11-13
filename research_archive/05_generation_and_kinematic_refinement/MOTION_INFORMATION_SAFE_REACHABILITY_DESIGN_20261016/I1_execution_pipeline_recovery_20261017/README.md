# I1 完整生产分块吞吐复核版本

父I1_execution_20261017未正式启动，原预检科学/恢复审核已通过但128-smoke外推58.4727min超过30min，全部原源/报告/失败保留。此新版本保留科学函数、数据、训练与审核内容，追加按实际2048保存chunk的完整forward/save/read/原聚簇/五motion/独立审核吞吐，预算再乘1.5大人口quantile/IO余量。只有新全预检与≤30min同时通过才独立启动；再次超限即BUDGET_NOT_LAUNCHED，不继续改规格。

I1两臂S/H×42/43/44，各25epoch，146输入/28632参数、FIT18909/outer1035/inner1856/CAL2879和DIRECT_READOUT_MSE不变。父2677sources含全部2560旧摘要继承，不覆盖；本目录source lock后不编辑科学文件。F0无solver且禁止安装，未启动；I1独立，G1没有F0见证不触发。
