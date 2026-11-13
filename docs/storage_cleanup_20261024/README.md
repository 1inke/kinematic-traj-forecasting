# 项目NPZ清理已完成（2026-10-24）

状态：**COMPLETE_VERIFIED**。用户明确要求直接运行清理脚本后，`RUN_CLEANUP.ps1` 已由助手实际执行，退出码0；并完成独立的计划/收据/保留文件复查。此前工具拒绝删除的状态已结束，旧记录保留于 `PRESERVATION_INITIAL_BLOCKED.json`。

## 实际结果

|项目|结果|
|---|---:|
|已清理的大NPZ源容器|601个，涉及16个历史方法目录|
|已移除源容器体积|22.58794 GiB|
|保留的无损共享档案（含索引和manifest）|15.47088 GiB|
|扣除全部本轮审查辅助文件后的净节省|约7.100 GiB|
|脚本结束时C盘可用空间|21.437 GiB|
|相对本轮脚本运行前增加的可用空间|8.851 GiB|

运行前空间已包含上轮新增的暂存档案，所以本轮前后可用空间增加约8.85 GiB，而相对未暂存前的净节省约7.10 GiB。磁盘还可能有其他应用写入，精确文件体积差见 `CLEANUP_RESULT.json`。

## 保留与验证

- 没有丢弃独有科学数组。完整NPZ的ZIP压缩成员和结构存入共享档案，相同数组只保留一份。
- 所有601个原NPZ均通过从持久化档案重新读取、恢复字节流后的完整SHA-256校验，原始SHA可精确还原。
- 12个非空删除批次；每个源文件删除前再次核对路径范围、体积和源SHA。601条删除收据与计划、数据库逐项一致。
- 51552个原有保留文件的存在性、大小、mtime和文件ID复查通过。这是元数据验证，不冒充所有权重内容SHA重审。
- 471份原分析/总结文档SHA-256保持不变；SQLite quick_check为ok。
- 代码、权重、唯一原始输入、split/ID/归一化统计、小型证据和失败日志保留。近期qoder/qoder_work、当前native与公共输入、重要诊断和G1上游保留。
- 没有重新训练、评测旧模型或改变科学结论、baseline、协议及自动任务。

每个受影响方法目录都有 `NPZ_STORAGE_NOTICE_20261024.md`，指出原NPZ的恢复方式。原历史SHA清单仍表示原容器内容；旧审核程序需要文件在线时，先恢复所需路径。

## 恢复原NPZ

保留整个 `docs/storage_cleanup_20261024/`，尤其是：

- `npz_archive/payloads.bin`
- `npz_archive/index.sqlite`
- `npz_archive/manifests/`
- `archive.py` 与 `review.py`

在项目根目录PowerShell中恢复某个方法：

```powershell
python .\docs\storage_cleanup_20261024\archive.py restore --path "MOTION_STATE_GOAL_PROCESS_DESIGN_20261014\M2_execution_20261014\"
```

可加 `--destination "<LEGACY_WORKSPACE>"` 恢复到其他盘，或用 `--all` 恢复全部601个源文件。需要额外空间；程序检查SHA并拒绝覆盖不同内容的既有文件。恢复只还原数据，不自动运行训练。

当前清理已完成，无需再运行清理脚本。`UNDO_STAGING.ps1`是上一阶段供原文件尚未删除时使用的撤销入口；当前应使用上述restore命令。

## 审查与清理记录

审查范围为192个含NPZ目录、3个根级NPZ文件，以及471份分析/总结文档。只把有明确停止/科学门失败依据的具体历史分支列入精简；结论不明、输入、预检与小型证据保留。科学门FAIL不解释成整个研究方向无效。

- `EXPERIMENT_DECISIONS.csv`：逐目录决定及报告路径。
- `APPROVED_SCOPES.json`：具体方法分支、停止依据、报告SHA。
- `ARCHIVE_PLAN.json`：候选、数组负载hash和原报告引用。
- `CLEANED_FILES.csv`：实际601个已清理文件、原SHA、恢复manifest和依据。
- `DELETE_RECEIPT_*.json`：实际删除收据。
- `FINAL_ARCHIVE_VERIFICATION.json`：601/601完整恢复校验PASS，source_removal_complete=true。
- `CLEANUP_RESULT.json`：独立最终复查与容量统计。
- `MANUAL_COMPLETION.json`：清理脚本原生输出记录；文件名沿用准备阶段命名，本次实际由助手执行。

本次只清理已审查的NPZ源容器；AV2原始数据和历史epoch权重未在此次操作中删除。
