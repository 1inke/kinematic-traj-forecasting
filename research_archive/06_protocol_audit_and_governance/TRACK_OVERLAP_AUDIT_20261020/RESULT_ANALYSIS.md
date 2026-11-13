# C1–C6 重叠审计：OPEN，不能正式冻结

范围：只读metadata、原split、原始focal世界坐标、上游split与HGB实际rows；无训练、model/optimizer、外部数据下载。独立脚本未导入旧训练/评价函数；由同一助手实现，不是外部独立AI审稿。完整输入/script SHA、可追溯ID和track见 `OVERLAP_REPORT.json` 与 `shared_tracks.csv`。所有读取输入前后SHA相同。

## 事实

- C1：四人口sample ID六对交集均0；当前科学人口总24,679窗，注册总28,791窗，其余并不自动加入训练。
- C2：FIT、inner、outer互相target-track不重叠。outer change253窗/68track中FIT已见track为0。FIT与original_CAL共享29track，涉及FIT906窗、CAL1,149窗。
- C3：inner/outer对FIT的已知同track 20索引完整区间交叠均0；CAL有457窗与FIT交叠，其中change130窗。报告同时保存未来12点任意交叠和完全落入单个FIT区间的两种口径，不能混用。123个原时间歧义窗口保持未知，不把未知当无交叠。
- C4：取整到1mm的20点世界坐标哈希，在六对人口均未发现同完整路径近重复；已知同track/sourceindex坐标冲突为0。发现1个track的窗口起始索引存在>20间隙，仅作screen；没有证明ID复用或证明绝无复用。source index不是未经核实的视频帧/秒。CAL命名同为students003，与其他人口同名空间。
- C5：outer来自FIT已有场景族，不是held-scene评测；分场景/组别数见JSON。
- C6：h3/v2实际split_indices训练集都覆盖outer全部1,035个身份窗；1,031窗有可核对时间交叠，4窗时间未知。原CAL2,879窗均同track且时间交叠。此为真实训练split文件证据，不假称重放了模型或恢复了训练时锁源。
- HGB实际三seed rows均4,096个训练窗口；与outer同track暴露0，但与CAL846窗同track、333窗已知时间交叠。KIN源码配方用当前FIT，因此与CAL交叠；每checkpoint训练IDs的完整manifest仍未证明，不能假装C6全清。Q0四cohort属于HGB既有训练/评估的诊断人口，不是本次新训练。

## 推断与处置

文档预定 C3>0 的处置已触发：新 S3 前必须建立track互斥的v1.1提案，作者审核后冻结；当前不改旧split、不重写历史结果。CAL不是未接触的独立盲集，上游native/encoder也不能在现outer上称整个模型未见验证。只修后续newhead FIT–outer互斥，不会自动消除F16先前见过outer的问题。

## 未证明

target-disjoint不等于邻居/context/time独立。1mm全路径哈希无匹配不排除部分重复或不同时间采样的同一人。KIN源码配方不替代每模型真实运行来源。此次没有完整上游重新训练，也没有证明轨迹不可预测、共享head是唯一根因或任何方向已被否定。

作者审阅状态保持OPEN；脚本不得自行将其改成CLOSED或把协议标为已冻结。
