# 实施核对与交付边界

本次为2026-10-23人类“开启三个文件夹、补充好代码文件”的代码准备，未授权/未触发新训练。原合同仍是上位依据，不修改原文。

## 对照与保护检查

- A：固定两基底、FIT-only非中心SVD、符号零阻塞；CV同初态/同82008参数配对，只换固定投影。筛查失败不解释成不可达、不自动训练。
- B：继承旧HistogramPrior接口并覆写不符合新合同的256桶概率/查询逻辑，旧`src/f16/scene_prior.py`未改。257桶、稳定top4、track purge、同序列已完结时间、冷启动非同序列库均有代码和合成反例；没有场景图像或模型。源成员用bitset保存，避免巨大常驻Python列表。
- C1：detach软责任的导数测试`w/B>=0`，退火全592步，原endpoint线性排斥不变；不叠相对P惩罚。
- C2：确实98552参数，零W2/零head分阶段激活；同时测试隐藏槽和完整queries的跨槽Jacobian（iso严格0、mix非0）、置换等变、amax并列均分、双精度参数FD双epsilon。不是把网络缩回82008以假造等容量。
- shared：forward只接受210features；FIT/inner白名单；旧A固定retained/rescued/lost/neither四人口不随新臂重定义；同candidate质量、GTfree complete-link、全K20五motion、J/Q各自槽位、旧源/完成输出SHA核验、全RNG断点和单worker byte-lock均提供。

## 三轮同助手反证（不是外审）

1. **设计误读风险**：没有把“补代码”解释成“训练三个方向”；A/B零训练阶段与C1/C2单项训练严格分开。无总自动启动器。
2. **伪PASS风险**：合成fixture不等于实际GPU/FIT源审计。`CODE_VERIFICATION.json`显式标actualGPU/实际screen/probe/training/OS task全false。C2缺整模测量无法PASS。B缺真实坐标/索引来源不能猜造通过。
3. **科学外推风险**：L2投影与训练可达性不同，endpoint潜力与完整路径不同，GTfree有效好簇不是越多越好；局部机制门与完整G_base分开。旧F16见过inner和重复开发限制保留，不声称独立泛化或唯一根因。

## 仍须在实际执行阶段完成

- 实际全继承源SHA核验与新execution锁源；未知源变化不能覆盖旧digest。
- A全量离线筛查、B真实world/time records与全量探针；目前均未运行。
- 单项训练另授权后实际GPU8、合法梯度、双方513FIT+12inner两epoch全RNG恢复、原loss/控制复现、保存128完整聚簇motion审核与吞吐/资源估算。
- C2整模MAC和冻结encoder+decoder的真实batch1/256延迟/峰值；当前代码提供decoder测量，不假定它满足整模要求。
- 正式结果出来后读取全部seed/组/失败与CI再写分析。代码交付阶段没有实验结论和ETA，不创建调度跟进。
