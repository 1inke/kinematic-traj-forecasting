# Prelaunch implementation provenance

First preflight stopped before GPU tests because the AST-only old loss reader used the Windows default GBK codec on UTF-8 source. Exact failed preflight.py and traceback are in failures/preflight1_utf8. Corrected only the explicit text encoding; the old loss source was not modified. The failed process imported the earlier training diagnostics implementation; subsequent logging extensions were not used in that failed process. Do not label a later copied training file as the exact failure-time loaded module.

Before the successful full rerun, the throughput estimate explicitly includes both native reference full audits in addition to six model-population audits. Epoch diagnostics record per-batch clipping/counts and fixed1024 teacher responsibilities. No scientific formula, tolerance, epochs, threshold or data population changed. No production worker had started.
