# 两方向联合训练：完整保存结果待人类阅读

固定12run/240epoch/162960更新，26人口。所有12final SHA锁后才开放dev评价；原科学源码未改。
原F16+h3仅在existing方向冻结；CVAE为新prior/posterior/decoder/DSF联合训练，共用冻结观测context。

|run|dev change good/960|native good|rescue|lost|ADE|FDE|
|---|---:|---:|---:|---:|---:|---:|
|existing_accuracy_only_seed42|371|350|92|71|0.259238|0.445777|
|existing_accuracy_dpp_seed42|373|350|92|69|0.258847|0.443534|
|existing_accuracy_only_seed43|368|350|92|74|0.259157|0.443892|
|existing_accuracy_dpp_seed43|368|350|87|69|0.258478|0.441311|
|existing_accuracy_only_seed44|360|350|84|74|0.258938|0.444999|
|existing_accuracy_dpp_seed44|366|350|85|69|0.258440|0.442450|
|cvae_accuracy_only_seed42|379|350|113|84|0.262133|0.470145|
|cvae_accuracy_dpp_seed42|374|350|116|92|0.267698|0.482797|
|cvae_accuracy_only_seed43|378|350|119|91|0.264222|0.466421|
|cvae_accuracy_dpp_seed43|371|350|114|93|0.265028|0.471778|
|cvae_accuracy_only_seed44|362|350|113|101|0.267303|0.474007|
|cvae_accuracy_dpp_seed44|384|350|119|85|0.264973|0.472654|

全部96seed/split/group和原五motion/保护门见ALL_GROUPS_REVIEW与PROTECTION_GATES。
固定family30所有普通/校正/fixedseed CI完整保留，不用单个阳性替代原门。
ARS DA2：加入监督不保证保留旧成功；CVAE重构与推理prior之间仍可能有差距，核所有救回/丢失与运动代价。
ARS DA3：两结构的容量/可达集和共享旧观测encoder不同，不能声称纯结构因果、三独立whole-model泛化或SOTA；同助手检查非外审。
成本为NOT_MEASURED_RAW_STREAM，不作部署PASS；场景/联合地图臂未执行，不当阴性。
不自动升级baseline或补扫权重/LR/epoch/额外变体，所有模型/失败/日志/自动记录保留。
