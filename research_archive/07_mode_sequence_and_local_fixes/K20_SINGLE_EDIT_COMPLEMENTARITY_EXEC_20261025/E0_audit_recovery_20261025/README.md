# E0保存审核的有界内存恢复

父E0 worker38480在2026-10-25 12:46:21资源STOP：实际保留private峰4,209,709,056bytes，超过1.3×3GiB=4,187,593,113.6bytes；当时可用physical14,632,505,344/commit17,916,592,128，不是机器没有足够余量，但不能事后洗掉超估。

原因：独立review.direct返回paths[oracle]视图，每个小20×12×2路径保留其背后401×20×12×2母数组。audit_seed逐窗把视图存入oracle_paths，约每128窗增加197MB。修复仅在返回边界对已选oracle路径copy，释放不再需要的401路径母数组；所有数学计算、mask/标签、oracle排序、2e-12数值容差及1.01科学门完全不变。

父科学源/FAILURE/newstderr/全部三seed枚举和done保持只读。继承父514全部digest，扩入父完成输出和失败来源。本恢复只重做全部4,915,200保存动作算术审核和完整oracle质量/聚簇/运动学/源与输出终SHA，无重新枚举、生成器forward、训练或新formal。

预检须真实证明原返回shares_memory母数组而修复返回不保留母数组、数值逐位相同，全部第一128以及故障邻接块等共512实际保存窗口读回独立审核、资源保留峰与有界增长、坏source/outputreader拒绝。资源仍E3GiB、1.3动态remaining+Windows retainedpeak，不inflate估计。实际吞吐后才恢复sole OS worker/一次startup/一次elapsed50%+完成节点。

同助手反证DA1：资源错误不改变E0科学门；父枚举/初步GATE不算完整科学结果，所有动作审核完成才判定E1。当前无新训练授权扩张、不动任何旧PAUSED任务。不是外审。
