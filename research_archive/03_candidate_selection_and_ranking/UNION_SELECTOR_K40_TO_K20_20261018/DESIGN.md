# UNION_SELECTOR_K40_TO_K20_20261018

Date: 2026-10-18

## Question
Can we select (or fuse) **K=20** from the native∪KIN pool (**K=40**) and keep a large fraction of the union coverage lift (~+9.4pp vs native), under motion-aware constraints?

## Pool
Reuse KINEMATIC_DECODER_SPEED_CLAMP_20261018 KIN checkpoints (seeds 42/43/44).
Per sample: concat(native_20, kin_20) → 40 candidates. Same ETH-UCY outer (N=1035).

## Baselines (report all)
1. native-only K20
2. KIN-only K20
3. union K40 (oracle pool upper bound)
4. Oracle K20 (GT-aware ceiling: keep ≥1 good cand if any in pool → cov = union)
5. Oracle minADE-top20 (GT-aware reference)

## GT-free selectors (real gate)
- strat_fps_10_10 / 12_8 / 15_5: stratified endpoint FPS within each family
- global_fps_20: FPS on all 40 endpoints → 20
- swap_redund_k: start from native; swap out most-redundant native for farthest KIN (k=4/6/8)
- motion_strat_12_8: same as strat FPS but drop high per-cand speed outliers first
- it_rank_fps (cheap): FIT-trained logistic score (is_good) + stratified FPS among top

## Gates (pre-registered)
- **HIT**: best GT-free mean outer cov ≥ native + **4.0pp**, and selected-set motion gate PASS (same 1.01× native stats), ≥2/3 seeds individually ≥ native+4pp OR mean≥+4 and all 3 motion PASS
- **STRETCH**: mean ≥ native + **7.0pp** OR within **2.0pp** of union mean
- Oracle reported separately; does not count for HIT

## Non-goals
Do not modify MODE_PRESERVING/P1. Do not resurrect MODE_DIVERSE as main.
