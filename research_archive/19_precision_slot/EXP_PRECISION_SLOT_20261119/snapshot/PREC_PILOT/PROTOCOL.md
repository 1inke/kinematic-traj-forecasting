# PREC-PILOT

32 轮，seed42，C0 fit 21,719 窗、dev 4,753 窗。对照复用 EXPLORE_FAST 的 B0@32 和原 E3_self@32，需核对源合同、数据散列、最终模型与输出散列、训练更新次数、每轮批次哈希。两次新训练从 seed42 初始化，沿用原 batch64、AdamW、反射与速度增强、EMA 0.995 和 32 轮固定调度，最后一轮评价。单路 CUDA，完整轮次保存恢复。仅 C0。

Prior 与 Prior_self 都把固定索引19的第20槽设为 CV 基准。该索引是预先固定的实现约定，未用 dev 选槽。模式特征、原 proposal_delta、原 scale×0.75 的轴向残差范围与 score_head 均复用。三锚点 CV 基准为最后观测位移乘 4/8/12，训练增强之后仍在对应的目标坐标系算该量。原 proposal_delta 零初始化，因此该槽从精确 CV 开始。其他19槽原构造不变，总参数量与各自对照相同。Prior_self 继续使用原 E3_self 对 20 个候选的第二次修正，额外轴向限幅 scale×0.30，包括 CV 槽。原 E3_self 的集合中心化输入和对角注意力照旧。

沿用原 B0 训练目标。用 B0/E3_self 保存的预测另外做不训练的最低分槽 CV 替换，继承原 logit。训练版报告 CV 槽成为 minADE、minFDE、联合达标以及 Top1 的频率，同时报告独占联合覆盖。评估 minADE/minFDE、同轨迹 ADE≤.2 且 FDE≤.4 的联合覆盖、Top1 及 stable/transition/change、转向/减速停止和机动分组。

单 seed 的初筛标准是 Prior_self 相对 E3_self 的 minADE 和 minFDE 均改善至少 1.5%。Prior 相对 B0、Prior_self 相对 E3_self 和 B0 均检查 Top1 FDE 相对恶化≤2%、change/转向/减速停止的 minFDE 恶化≤1%、原四群体的 GT 运动学距离增量≤.05 与尾部超限率增量≤1 个百分点。另报 Prior_self 相对 Prior 的贡献、对两份零训练替换对照的净改善、候选覆盖损失与新增。初筛数值是这次开发实验的继续条件，仅用于决定是否值得增加配对 seed 与预算。
