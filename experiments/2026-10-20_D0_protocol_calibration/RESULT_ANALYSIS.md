# D-0 工程回放与功效近似

## 1. 登记摘要

仅CPU、0正式训练/0GPU小时。旧EXP03 A/B各三seed outer保存输出；native是同一冻结参考，复制不等于三个独立native训练。无original_CAL指标访问。协议/输入/输出SHA和实际运行记录见 `RESULT.json` 与各 `*_AUDIT.json`。

## 2. 判定适用性

本轮为工程与精度校准，非新S3，**不对历史实验追溯给新协议的四类科学判定**。工程回放/独立审计完成，不等于A/B新方法PASS。

## 3. 主人口校准

outer change253窗、68track；20,000 draws，PCG64(20261020)，每draw track→seed，窗口加权后seed均值。普通95%区间，无事后Holm“校正区间”伪称。

| 旧保存contrast | Δcoverage(pp) | 普通95% CI(pp) | 近似80% MDE(pp) |
|---|---:|---:|---:|
| A−native | −0.659 | [−5.634, 3.867] | 6.823 |
| B−native | +0.659 | [−2.464, 3.931] | 4.582 |
| B−A | +1.318 | [−2.545, 5.469] | 5.738 |

MDE由实测bootstrap SE乘 `(z.975+z.8)`，是双侧α=.05正态近似；不是实证power模拟、不适用于任意方法，也未考虑未来多arm校正使功效进一步变化。观察到的contrast决定SE；不能挑最小MDE当全项目保证。

## 4. 护栏与诊断

新框架保存Q-winner同路径ADE/FDE，区别旧报告minADE/minFDE。历史已使用的loss/运动标准不改；此处不将新G4=10%倒灌旧1%门以宣称历史成功。所有完整outer的4组质量、聚簇、运动学与修正量均另存；7套路径（native+6arms）全字段独立重算、6份旧报告共同字段回放通过。

## 5. 事实

native outer625/1035、change110/253；原NPZ的ID/native/GT与冻结缓存EXACT。三contrast区间均跨0。旧读取输入和checkpoint/history前后SHA不变；未加载模型。运行资源与耗时由RESULT实测记录，不用单元测试或预检峰值冒充生产训练峰值。

## 6. 推断

68track人口下当前小效应难稳定判别；这些例子的近似MDE均超过4pp参考。是否改MDE目标、采用5seed或S4汇总由作者在新登记前确定；不能本轮解封CAL补到显著。

## 7. 未证明

区间跨0不等于等效、证伪或没有信息。旧A/B使用不同阻尼项不等于新设计A场景先验/B联合生成。旧checkpoint缺optimizer/完整RNG，不因新接口出现就成为exact-resume合格。

## 8. 偏离与下一步

这是历史保存回放、不是同批fresh control；Holm CI具体约定仍待作者冻结。C3/C6处置未关闭前不启动新S3。先提交v1.1分区与上游训练暴露处理方案，再完成A0或B原生adapter和S1实际GPU验收。

## 9. 同助手反方复核（非外审）

DA2：功效数字可能被挑选？三contrast全部报告，普通区间不冒充Holm，固定68track无window伪重复。DA3：独立审计仍可能共同相信错误标签？新增审计核对冻结GT/native/group的ID与SHA；仍须承认旧数据本身的上游暴露。以上只界定结论，不制造新科学阳性。
