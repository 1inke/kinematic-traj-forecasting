# U/D四个单因素pilot：保存证据复核分析

结论：执行与数值审核通过，但四臂均未通过完整pilot门。U-Delta没有证明新息更新优于原B0；D有小幅误差线索，但差分版本没有建立对普通门控的总体优势，不自动推进三seed/V/F。

四fresh seed42固定8轮，共32epoch/10880更新。worker实耗29.837分钟，原规划29.927分钟；数值预检/成本/额外回归加worker实测合计36.464分钟，超过上位全流程30分钟预算估计。该偏差如实保留，不将worker单独未超30分钟冒称全流程达标。

|模型|总体ADE(m)|总体FDE(m)|总体覆盖|change ADE(m)|change覆盖|change Top1条件召回|
|---|---:|---:|---:|---:|---:|---:|
|B0|0.204214|0.361499|2179/4753 (45.845%)|0.263121|350/960 (36.458%)|30.286%|
|U-GRU|0.205944|0.365229|2192/4753 (46.118%)|0.262322|351/960 (36.562%)|29.345%|
|U-Delta|0.205313|0.362940|2204/4753 (46.371%)|0.263712|340/960 (35.417%)|29.706%|
|D-Gate|0.203699|0.360808|2195/4753 (46.181%)|0.262033|352/960 (36.667%)|26.705%|
|D-DiffGate|0.204412|0.362326|2176/4753 (45.782%)|0.261880|353/960 (36.771%)|28.895%|

## 1. U：新增分支可训练，但没有兑现change改善

U-Delta总体净多25成功窗，其中stable净多31、transition多4、change反而少10；change原350变340。相对B0总体ADE恶化约0.54%、change ADE恶化约0.22%。相对GRU总体ADE略好，但change更差、少11成功窗。不能用stable增益代表机动预测改善，也不能把这次阴性归为永久零梯度；三阶段实测与恢复门已经通过。
U-Delta dev观测机动/非机动的innovation mean约1.808/1.794，beta约.38695/.38876；没有显现更强的机动写入均值。这是固定分层下的描述，不证明门完全没学或否定所有新息结构。GRU和Delta的状态范数不是同一量纲指标，不能直接比大小宣胜。

## 2. D：普通门控是较温和的线索，差分特异优势未立

D-Gate总体ADE约下降0.25%、change ADE约下降0.41%，总体覆盖净多16、change仅多2。D-DiffGate change ADE约下降0.47%、change多3，但总体ADE略恶化、总体少3、stable少11。与普通Gate直接比，差分总体ADE更差约0.35%、总体少19个成功窗，change只多1个；不能挑change微小优势称差分机制被验证。
B0 change条件Top1召回106/350=30.29%；D-Gate94/352=26.70%；D-DiffGate102/353=28.90%。候选交互并未在这次pilot解决排序漏选。不同模型已覆盖集合不同，条件比率与无条件Top1成功数同时保留。
D-DiffGate总体effective-good约.549高于B0约.493，但change约.424低于B0约.430，且总体覆盖下降。更多好候选主要集中于已较容易的stable人口，不等于新窗口成功或语义模式理解。固定128注意力负权约1.588%、行和约.5，只证明实际启用差分算子，不证明去噪/互补因果。

## 3. 净增不能隐藏旧成功丢失

相对B0总体 rescue/lost：U-GRU125/112、U-Delta122/97、D-Gate98/82、D-DiffGate71/74。change分别25/24、19/29、20/18、15/12。所有旧成功分母及逐ID的J裕量伤害保存在SUPPORT_EXCHANGE.csv和ALL_GROUPS_REVIEWED.csv。此处参考是匹配的新B0，不将其伪称旧native同人口。

## 4. 尾部门没有通过，不是只有零分母阻碍

所有arm的超阈率增加均在+1个百分点范围，但正超出量p95有真实恶化：D-Gate stable速度约+2.58%；D-DiffGate stable速度约+2.84%、overall速度约+1.50%、overall接续jerk约+1.59%、change接续acc约+1.23%，均超过1%幅度门。U还存在更大的acc正超出幅度恶化。即使将无正jerk样本的NOT_ESTIMABLE单元暂不用于判定，也不能得出尾部门通过。
B0若没有正超出样本，正超出条件p95相对比无法估计；保持NOT_ESTIMABLE_NO_FLOOR，不把0/0补成PASS，也不把它称成实际出现运动违规。fit-GT阈值是经验参照，不是生理硬上限。完整五pooled与junction各组表保留。

## ARS DA2 / DA3与统计边界（同一助手角色，非外审）

No cherry-picking: UDelta overall net+25 mainly stable+31 while change-10; DDiff effective-good rises mainly stable but coverage falls3 overall. DGate modest error gains exceed no significance criterion. Keep all rescue/lost IDs and physical-tail failures.
No full-screen arm; relative tail magnitude failures exist even excluding unestimable zero-jerk cells. Do not attribute stop only to 0/0 handling, infer permanent zero gradients, or promote V/F on tiny single-seed gains. Extra capacity/optimization and signed-attention reweighting are alternatives to mechanism explanation. Full workflow timing estimate also exceeded budget when measured preflight components included.
One seed42 per arm, same dev4753 correlated windows/248tracks, change960/102tracks. Descriptive paired points only; no p values or confirmatory CIs. No new independent whole-scene/generalization or causal-intent claim.

下一步：本请求只分析，不新增训练。没有足够证据按原门自动补43/44或进入V/F。D-Gate可保留为弱线索，但不能事后将对照改名主方法；U和差分D的具体假说本次未获支持，不外推所有运动状态/协同方法无效。所有旧模型、代码、失败、日志和自动记录保持，baseline不升级。

复核291个去重来源/输出SHA全部一致；全部8科学人口保存算术/原完整complete-link+pdist/运动双实现收据齐。此次只读已保存数据、重数事件与核SHA，不训练、不全模型/梯度回放、不重复全聚簇，也不查已完成PID/task。
