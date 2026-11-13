# 同科学任务的预算测量复核

父预检完整报告保留：train nominal260.4457s；128窗口含文件创建、GC、压缩读回与两次完整指标的smoke各1.5192/1.6551s，外推保存审核2552.8417s，规划58.4727min，未启动生产。

128是必做的真实保存正确性smoke，不是实际保存chunk大小（生产2048）。新版本仅追加固定FIT前2048窗口两臂完整生产分块与独立保存算术审核的吞吐，使用固定非零throwaway初始化、不保存benchmark模型、不挑科学效果。所有25epoch、24science+8reference、全ID、GTfree聚簇和全部审核均不减少。按197432窗口的完整任务外推，并在pipeline额外乘1.5覆盖更大人口排序、quantile和IO，整体仍乘1.2并计训练/两次所有bootstrap/source余量。预算门仍30min；原58.5预测不删、不称科学失败，若新实际预测仍超门则停止不启动。

所有生产科学模块AST保持；common只增加父源继承和新增报告锁，preflight只补真实pipeline计时，memory_policy只让required_available报告明确已执行的max(系统6GiB,1.3remaining)，判断本身不改。Windows名称仅隔离新任务。修复不是拆成长任务或放宽预算。
