# EXP-178A result analysis

## Decision

**STOP_SCENE_TOPOLOGY_TRAINING** (passed: `false`).

## SDD test result

| Metric | Baseline K20 | K20 + topology | Relative improvement |
|---|---:|---:|---:|
| minADE | 6.2355 px | 6.2355 px | 0.00% |
| minFDE | 12.6859 px | 12.6859 px | 0.00% |
| same-path FDE | 14.2229 px | 14.2229 px | 0.00% |

- Positive test videos: 0/17.
- Little subset gate: False.
- The topology assets never use validation/test trajectory labels; test futures are accessed only here for final audit metrics.
