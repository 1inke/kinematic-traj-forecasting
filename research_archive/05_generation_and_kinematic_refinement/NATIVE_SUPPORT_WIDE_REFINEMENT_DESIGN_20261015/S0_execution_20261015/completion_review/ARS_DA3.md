# ARS反方检查3：最终结论压力测试（同助手 inline，非外部审稿）

结论：PASS；停止S1，旧baseline/历史cycle11/总目标不变。

最强反驳：保守ray的失败可能是选择的安全子族几乎锁死，而不是共享newhead本身没有映射能力。
采纳该反驳作为结论边界：S0没有训练模型，不能判定loss、P2、共享干扰、scorer或真实运动理解根因。

- No cherry picking：报告所有4人口×4组×3端口48行指标，全部20slot救回计数均为0。保留A0all20放松上界、微小误差改善、stable少数大修正反例。
- No moving goalposts：严格同candidate ADE<=.2且FDE<=.4；FITchange4725/native1937/至少237净救回；全部五运动学literal1%不加epsilon，不换半径、方向或margin。
- No fictitious failures：生产worker与exit双SUCCESS，科学gate FAIL；最初预检端口标签KeyError及来源保留，不能混为训练或生产失败。没有S1训练，更无seed间泛化结论。
- No false independence：全部人口来自已见缓存，target-disjoint不等context-disjoint。无新seed/CI，无wholemodel未见泛化/SOTA/心理意图声明。
- No overclaiming audits：保存数据算术重构不是全部模型/梯度重放或外审；报告AI辅助并使用ARS进行范围核查。
- Missing evidence：其它可改变时间形状且符合原组内五统计门的可行生成仍未由本实验测试，不能自动宣布其有效或立即换新实验。
- Required action：执行STOP_SAFE_OPPORTUNITY_UNDER_REGISTERED_MAP；S1及任何后续自动训练不启动，等待新精确设计/人类决定，不用局部扫描绕门。

压力测试：删去A0上界参照仍0新增而S0门失败；反向命题“其它安全空间可能有效”仍可信；更换人群不能外推；本结果价值是阻止对近乎收缩到原生的映射支付六次训练成本。

