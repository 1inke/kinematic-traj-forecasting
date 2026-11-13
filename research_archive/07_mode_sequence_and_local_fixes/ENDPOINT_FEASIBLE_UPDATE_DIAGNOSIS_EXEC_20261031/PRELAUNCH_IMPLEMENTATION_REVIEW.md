# Pre-science execution refinement

No scientific proposals had run. V1 source is retained in
`preflight_v1_source_snapshot/`; its full math/resume/pooled receipts remain.

Before science, review found two implementation-completeness issues:

- Generic inherited checkpoint writer committed a file directly. V2 now writes
  a new `.partial`, flushes+fsyncs, then no-overwrite Windows renames into the
  committed name. Partial evidence is retained on interruption.
- V1 feasibility timer excluded model inference. V2 measures actual full128
  inference plus gates and three real forced-eight-rejection transactions.

V2 also makes the conservative additional disk inventory explicit (1.5GiB,
1.3 admission), without deleting any earlier file.
All mathematical/transaction/resume/independent full128 checks are rerun before
scientific proposals. O0 is already complete and is SHA-compatible read-only reuse.
There is no changed loss, alpha, population, gate, optimizer or model setting.
