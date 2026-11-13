# I1 实施：历史运动信息直接读出

人类2026-10-17明确批准按父目录有限F0/I1/条件G1设计实施并合理安排。当前本文件为实施说明，真实状态以runtime_state/PREFLIGHT/worker为准，不预填PASS或running。

唯一因素S末速度重复+零加速度 vs H实际7速度+6加速度。P296、native24、history26共146输入，同28632参数MLP与共同FIT统计；直接无界MSE读出，非候选生成成功。固定6fresh×25epoch，outer epoch25为primary；inner/CAL仅posthoc。无旧F16前向/解冻、新数据、teacher、keep、投影、winner或diversity目标。

F0已因既有运行环境缺少SOCP依赖且指南禁止安装而记未启动，不判无安全解；I1独立推进，G1仍需F0见证与I1信号双门。本实施不更改父设计或旧任何科学结果。每阶段含保存完整审核的实测规划不得超过30min。

所有新增仅本目录。采用原1.3任务峰值/Windows retainedpeak检查及父设计系统physical/commit各6GiB预留。分块保存、一次独立OS worker、一次startup与预计50%/完成节点，无高频对话轮询。
