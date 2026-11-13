# 运动学解码 + 速度/加速度软限幅（2026-10-18）

承接：`KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018`（三 seed 接近门控但 0/3 weak_positive）。

本设计号唯一新增假设：在显式积分路径上对 **速度与加速度做软限幅**（相对 FIT-native 参考），使 outer 运动学相对 native 倍率 ≤1.01，同时保持相对 BASE 的 coverage/ADE 优势。

不碰 MODE_PRESERVING / P1。
