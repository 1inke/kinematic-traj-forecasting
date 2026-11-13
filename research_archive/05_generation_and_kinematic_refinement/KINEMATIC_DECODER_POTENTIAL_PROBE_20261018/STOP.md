# 停止：潜力门未命中

- potential_gate_HIT: **false**（A=0/3，B=false，两变体皆然）
- ceiling_under_recipe: **true**
- best: LONG_RELAX mean outer cov **0.5646** vs native **0.6039**（约 −3.9pp）
- motion PASS: 3/3（两变体）
- 迁移 FIT↔outer: |gap|<1pp，migration_ok
- 加长 32 epoch + 轻度放松限幅 **未能**追上 native；与 8-epoch SPEED_CLAMP 覆盖几乎同平台
- mid-eval 峰值偶见 ~0.58（仍 < native），继续训反而略回落 → 不像单纯欠拟合
- BASE@32（参考）outer cov 0.574 但 motion FAIL（accel）
- 不碰 MODE_PRESERVING/P1

用户一句：加长训练和略松限幅后，KIN+clamp 仍卡在覆盖约 0.56（native 0.60），运动合格但冲不过原生覆盖——本配方上行空间有限。
