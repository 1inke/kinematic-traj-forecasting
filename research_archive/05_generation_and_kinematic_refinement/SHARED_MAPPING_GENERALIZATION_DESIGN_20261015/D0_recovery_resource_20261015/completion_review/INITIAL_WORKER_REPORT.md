# D0 finite-update diagnosis

Decision: D0_STOP_NO_U1

|seed|ordinary source dL|projected source dL|ordinary probe dJ|projected probe dJ|direction packets /8|ordinary lost|projected lost|gate|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|42|-0.001642853908|-0.001676479207|9.14263472e-06|9.765501678e-06|5|1.25|1.25|False|
|43|-0.001592764716|-0.001539606245|4.209459167e-05|3.291094681e-05|6|0|0|False|
|44|-0.001691138392|-0.001661077975|-1.437628334e-05|-1.633152075e-05|4|0.5|0.375|False|

All 3 seeds and all8 packets included. Same complete candidate ADE<=.2 AND FDE<=.4. Source and probe are previously seen FIT, track-disjoint but context may overlap. Temporary fresh-zero-moment AdamW is an intervention at final checkpoints, not original training resume or independent generalization.

Independent saved audit: PASS, jobs=219, maxerror=0.0. Old scientific sources and parent checkpoints SHA preserved. NoUpdate exact. All groups and step0/1/4 snapshots stored. No new CI.

ARS inline DA2 (same assistant, not external): local dot products cannot establish finite-step causality or generalization; gates use actual source loss/probe J and retained-good lost. Fixed bottom4/.20m position envelope and saturated final head may limit intervention.

ARS inline DA3 (same assistant, not external): D0 failure stops U1. D0 positive would only permit U1 own preflight/budget; no baseline/goal completion or revival of failed T1. Original major and five-motion1% remain.
