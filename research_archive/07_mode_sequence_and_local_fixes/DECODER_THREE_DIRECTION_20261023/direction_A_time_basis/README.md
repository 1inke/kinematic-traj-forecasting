# 方向A：时间结构化基底

状态：`CODE_PREPARED_NOT_LAUNCHED`。`basis.py`实现固定t=4/8/12 knot3、只FIT非中心SVD4、正交投影；`screen.py`完成GT单路径重构与三seed旧A全K20后处理、完整质量/聚簇/五motion/velocity_join/Jslot Ea及筛查门。选择和停止标准见 `protocol.json`，不扫rank或节点。

将来获准运行时，在项目根目录调用（E需要先明列inventory；下面仅是命令格式）：

```powershell
# CPU筛查：必须显式给新输出目录和估算值，当前没有执行。
python -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_A_time_basis/screen.py --output <A目录下的新screen目录> --estimated-peak-gib <实际估算E>
# 筛查通过且另获A-Train授权后，逐阶段准备/预检/训练，不自动串行：
python -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_A_time_basis/train.py --execution <新的训练目录> --prepare --screen <完成的screen目录>
python -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_A_time_basis/train.py --execution <训练目录> --preflight --estimated-peak-gib <E>
python -B qoder_work/DECODER_THREE_DIRECTION_20261023/direction_A_time_basis/train.py --execution <训练目录> --train --authorization <明确A单项授权JSON> --estimated-peak-gib <同预检E>
```

授权记录仅在真实新指令后保存，字段为 `package: A`、`direct_human_instruction: <真实原话>`，不能为当前代码准备请求伪造训练授权。A两臂82008、共享CV起点、相同旧统计/初值/8轮shuffle；历史A输出只辅助，不拿它顶替fresh控制。所有输出包含明确ID和源绑定，已有done只能完整SHA匹配后复用。
