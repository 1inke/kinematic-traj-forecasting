# 大型NPZ存储位置变更（2026-10-24）

本目录有18个历史输出NPZ进入经授权的无损精简清单，源容器合计0.2970 GiB。实际完成状态以项目根目录 `docs/storage_cleanup_20261024/FINAL_ARCHIVE_VERIFICATION.json` 和删除收据为准；本说明本身不证明迁移已完成。

代码、权重、分析报告、划分/统计输入与小型证据保留。已清理的原NPZ的全部字节保存在项目根目录 `docs/storage_cleanup_20261024/npz_archive/` 的共享负载、数据库和逐文件manifest中，没有丢弃独有数组。恢复后原SHA-256不变。

在项目根目录PowerShell中恢复本目录所需的原NPZ，再运行历史评测/审核脚本：

```powershell
python .\docs\storage_cleanup_20261024\archive.py restore --path "GENERATION_NATIVE_ANCHORED_BERNSTEIN_SERIES_20261008\"
```

可追加 `--destination "<LEGACY_WORKSPACE>"` 恢复到其他盘；需要相应空间。原历史SHA清单保持原样，记录的是原容器；文件不在原位时，应先恢复，不应改旧SHA或当成实验失败。精确文件与方法停止依据见 `ARCHIVE_PLAN.json`，删除收据见 `DELETE_RECEIPT_*.json`。

不要删除共享档案；它是这些输出的恢复来源。此操作只整理存储，没有重新训练、评测或改变科学结论。
