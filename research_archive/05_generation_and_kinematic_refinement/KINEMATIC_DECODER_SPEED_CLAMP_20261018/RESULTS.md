# 结果报告 SPEED_CLAMP round=r1

时间：2026-10-18T13:12:50+0900（Asia/Tokyo）

## FACTS

| seed | arm | coverage | minADE | motion | fails | accel_r | speed_r | weak_pos | stronger |
|---:|---|---:|---:|---|---|---:|---:|---|---|
| 42 | BASE | 0.4271 | 0.3143 | FAIL | speed_p95,accel_mean,jerk_mean | 1.801 | 1.028 | - | - |
| 42 | KIN | 0.5652 | 0.2103 | PASS | - | 0.913 | 0.970 | True | False |
| 43 | BASE | 0.4058 | 0.3316 | FAIL | speed_p95,accel_mean | 1.371 | 1.168 | - | - |
| 43 | KIN | 0.5652 | 0.2047 | PASS | - | 0.932 | 0.970 | True | False |
| 44 | BASE | 0.4164 | 0.3242 | FAIL | speed_p95,accel_mean | 1.732 | 1.208 | - | - |
| 44 | KIN | 0.5633 | 0.2109 | PASS | - | 0.956 | 0.970 | True | False |

weak_positive **3/3**；stronger **0/3**；KIN motion PASS **3/3**
coverage 增益均值 14.81±1.07 pp；minADE 比 0.646±0.026

## STOP: strong_good_result=True

## INFERENCE
Round r1: weak_positive 3/3, stronger 0/3, KIN motion PASS 3/3。已达比较强的好结果，可停。
