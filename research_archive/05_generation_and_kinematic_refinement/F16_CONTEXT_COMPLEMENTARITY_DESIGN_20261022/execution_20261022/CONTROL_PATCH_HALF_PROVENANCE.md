# Halfway control patch provenance

First control-only patch was rejected before application because two ledger hunks were supplied in reverse source order (experiment entry before its preceding monitor entry). Read-only control inspection confirmed none of ledger/memory/runtime-control changed. Corrected hunk order, retaining the same actual single halfway check and timing. This is a control editing error, not a worker/science/resource failure; no locked source or experiment data was changed and no process check was repeated.
