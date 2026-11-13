# EXP-180B Deployment Backend Audit

固定模型：EXP-173 final F16 + H3 + Joint-V2 + original score + K20。

## 判定标准

- ONNX 相对 PyTorch 的 minADE/minFDE 变化均小于 0.2%。
- 中位延迟降低至少 20%，或吞吐率提高至少 25%。

## Runtime

| Profile | Backend | Median ms | P95 ms | Throughput/s | Median gain | Verdict |
|---|---:|---:|---:|---:|---:|---|
| sdd_n16 | B0_pytorch_fp32 | 12.7694 | 13.0537 | 78.31 | - | REFERENCE |
| sdd_n16 | B1_ort_cuda_fp32 | 4.7859 | 4.8647 | 208.95 | 62.52% | PASS |
| sdd_n16 | B2_ort_cuda_fp16 | 5.1781 | 5.3586 | 193.12 | 59.45% | FAIL |

## Correctness

| Profile | Backend | ΔminADE | ΔminFDE | Candidate max abs | Verdict |
|---|---:|---:|---:|---:|---|
| sdd_n16 | B1_ort_cuda_fp32 | 0.0130% | 0.0506% | 0.004608 | PASS |
| sdd_n16 | B2_ort_cuda_fp16 | 0.4390% | 0.3451% | 0.013438 | FAIL |

## 自动结论

存在通过 EXP-180B 门槛的部署组合：sdd_n16/B1_ort_cuda_fp32。正式保留前需结合两个 profile 综合判断。

说明：ONNX 导出对固定尺寸 `EyeLike` 与旧导出器的 `atan2(0,0)` 展开做了等价兼容处理。
