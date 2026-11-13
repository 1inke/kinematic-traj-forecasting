# 设计：原生 ∪ 运动学再生 并集篮子

日期：2026-10-18

## 假设
原生 K20 与 KIN+clamp 再生篮子互补：并集覆盖可超过任一单侧。

## 方法
复用 `KINEMATIC_DECODER_SPEED_CLAMP_20261018` 的 3-seed KIN checkpoint；
在 outer 上拼 `concat(native, kin)` → 40 候选，算覆盖与 kin_only / native_only / both。

## 预注册门
- **strong_complement**：KIN motion PASS 且 union−native ≥ 1pp
- **weak_complement**：KIN motion PASS 且 kin_only > 0
- **gate_HIT**：≥2/3 seed 达 strong

## 说明
并集含原生时 union≥native 几乎平凡；关键看 **lift** 与 **kin_only**。
