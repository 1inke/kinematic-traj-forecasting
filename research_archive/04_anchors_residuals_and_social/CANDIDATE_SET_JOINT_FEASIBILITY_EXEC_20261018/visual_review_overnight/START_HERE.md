# 过夜结果可视化（接入候选可视化诊断工具）

- 样本：ETH outer seed42，N=1035，每例 K20（从 native+KIN 的 K40 池选出）
- 模型切换：native / v1 / k1_I / k1_S / k0c / deploy_A / deploy_B
- 橙线=组内最高分槽（scores40 或原生 logits），不是 GT 选择
- k0c 集合本身是池可达 oracle 参考；deploy_B 是去 pred_gate 的调整变体
- **坐标系**：世界米制 + 按场景显示重映射——students/zara 画 (x,y)；hotel/eth 画 (x,-y)。**不是**全场景通用「视频同向上」。详见 COORD_FRAME.md

## 打开方式

1. 双击运行 `START_SERVER.bat`，或在本目录执行：
   `python -m http.server 8765`
2. 浏览器打开：http://127.0.0.1:8765/index.html
3. 建议先看 model=`k1_S`，再对比 `k0c` 与 `deploy_B`
4. hotel 例：搜 ID `740`，确认轨迹上下与视频一致

数据由 `build_overnight_visual.py` / `patch_display_axes.py` 生成；不改写旧 P1。
