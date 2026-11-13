# Endpoint-horizon idea: similar samples to 24102

**Idea (user, 2026-10-19 Asia/Tokyo):** For 24102, models put the 12-step endpoint in front of a person-wall while GT goes around downward. Suggestion: consider a farther future endpoint (beyond the current prediction window) so the generator/selector can prefer more flexible detours.

**This note:** operational screen for endpoint-large cases on ETH outer seed42 visual set (N=1035). Does **not** change any P1 protocol.

## Screen definition

- Flags (need >=2, or composite score >=1.2):
  - minFDE_native >= 0.8 m
  - minFDE_k1_S >= 0.8 m
  - top-logit native endpoint vs GT endpoint >= 1.0 m
  - GT endpoint vs constant-velocity endpoint |lateral| >= 0.8 m
  - best-minADE native endpoint vs GT endpoint >= 0.8 m
- Prefer: university/students + neighbors; exclude 24102; at most 5 per sequence.
- Time: video_sec ≈ annot_frame * 0.4 (frame×10/25). Observation start = start_frame; **t0** = start+7; **pred_end** = start+19.

## Reference: 24102

- sequence `students003`, ped `102`, start_frame `144`
- obs≈57.6s, t0≈60.4s, pred_end≈65.2s, neighbors=7
- minFDE_native=0.936, minFDE_k1_S=0.936, top_end_gap=2.543, gt_vs_cv_lat=2.734

## Selected similar samples (n=18, raw flagged=336)

| sample_id | sequence | start_frame | t0≈s | pred_end≈s | n_nei | minFDE_n | top_end_gap | lat_vs_CV | why-like |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 24063 | students003 | 142 | 59.6 | 64.4 | 4 | 2.392 | 6.74 | 1.927 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域；校园广场 |
| 20242 | students001 | 319 | 130.4 | 135.2 | 4 | 0.44 | 5.338 | 4.163 | top终点远离GT；相对匀速大横偏；多人邻域；校园广场 |
| 9433 | students001 | 6 | 5.2 | 10.0 | 8 | 1.239 | 5.072 | 1.787 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域；校园广场 |
| 24563 | students003 | 166 | 69.2 | 74.0 | 9 | 1.277 | 4.955 | 2.754 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域；校园广场 |
| 25936 | students003 | 224 | 92.4 | 97.2 | 9 | 0.653 | 4.264 | 4.944 | top终点远离GT；相对匀速大横偏；多人邻域；校园广场 |
| 23454 | students003 | 111 | 47.2 | 52.0 | 8 | 1.105 | 4.864 | 2.52 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域；校园广场 |
| 21477 | students003 | 30 | 14.8 | 19.6 | 9 | 0.663 | 4.819 | 2.958 | top终点远离GT；相对匀速大横偏；多人邻域；校园广场 |
| 20275 | students001 | 320 | 130.8 | 135.6 | 4 | 0.991 | 4.295 | 3.469 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域；校园广场 |
| 20013 | students001 | 312 | 127.6 | 132.4 | 9 | 1.458 | 3.888 | 3.724 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域；校园广场 |
| 9332 | students001 | 4 | 4.4 | 9.2 | 8 | 0.088 | 3.86 | 1.556 | top终点远离GT；相对匀速大横偏；多人邻域；校园广场 |
| 2923 | crowds_zara02 | 44 | 20.4 | 25.2 | 5 | 1.803 | 4.445 | 3.325 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域 |
| 8558 | crowds_zara03 | 348 | 142.0 | 146.8 | 5 | 0.356 | 4.194 | 3.393 | top终点远离GT；相对匀速大横偏；多人邻域 |
| 1560 | crowds_zara01 | 199 | 82.4 | 87.2 | 3 | 0.82 | 3.878 | 3.223 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域 |
| 6445 | crowds_zara02 | 721 | 291.2 | 296.0 | 4 | 0.456 | 2.979 | 3.385 | top终点远离GT；相对匀速大横偏；多人邻域 |
| 8538 | crowds_zara03 | 344 | 140.4 | 145.2 | 5 | 0.993 | 3.255 | 2.921 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域 |
| 6802 | crowds_zara02 | 759 | 306.4 | 311.2 | 9 | 1.059 | 2.123 | 3.12 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域 |
| 5840 | crowds_zara02 | 630 | 254.8 | 259.6 | 9 | 1.108 | 3.023 | 2.056 | top终点远离GT；K20也盖不住末点；相对匀速大横偏；多人邻域 |
| 5831 | crowds_zara02 | 629 | 254.4 | 259.2 | 9 | 0.277 | 2.993 | 1.216 | top终点远离GT；相对匀速大横偏；多人邻域 |

## How to watch

1. Open the matching ETH/UCY video (students003 / students001 / uni_examples / etc.).
2. Seek to **t0≈** for prediction start; **pred_end≈** for current 12-step horizon end.
3. Ask: does a farther horizon (e.g. +2–4 s past pred_end) make the go-around / intent clearer than the 12-step endpoint alone?

Companion JSON: `ENDPOINT_HORIZON_IDEA_SIMILAR_SAMPLES_24102.json`
