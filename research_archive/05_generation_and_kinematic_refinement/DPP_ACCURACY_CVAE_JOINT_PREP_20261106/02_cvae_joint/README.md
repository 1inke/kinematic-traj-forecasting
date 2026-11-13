# 方向二：新CVAE与DSF联合训练

入口`run.py`默认只打印计划。新prior/posterior/decoder/DSF从零联合训练；同一冻结观测64D context保留，旧h3轨迹生成结构/score移除。posterior只训练时用未来，generate(bundle)只使用观测上下文。

两臂相同Gaussian重构+.01KL+同slot ADE+.5FDE，唯一差别是否加-.1ECkin/20，三seed配对初值与数据序。当前allow_training=false，未预训练、未联合训练、未启动任何进程。不是用GT posterior采样伪造推理性能，也不宣称整个模型变轻。
