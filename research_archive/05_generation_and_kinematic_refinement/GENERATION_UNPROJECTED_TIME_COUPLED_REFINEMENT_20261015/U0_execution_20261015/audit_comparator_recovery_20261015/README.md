# U0 saved-report comparator infrastructure recovery

First audit recovery11480 ended21:09:27.283244HK with TypeError in original
common.compare: BOOTSTRAP.rows is a list of dictionaries, but fallback numeric
conversion cannot handle dictionaries. Trace reached the bootstrap comparison
after28 saved-job and6 checkpoint loops; no final ARITHMETIC_AUDIT/PRESERVATION
exists, so no final scientificPASS is claimed from loop progress or traceback.
Resource failure did not occur: recorded peak private2472628224bytes and minimum
physical20424343552/commit25029771264bytes. Original failure/code/log retained.

Only repair: compare lists/tuples recursively, require exact lengths/dict keys,
retain exact string/bool/None semantics and original numeric2e-12 tolerance.
No statistic/cluster/motion/quality/ID/seed/threshold/CI family or saved model
change. Actual nested-list TypeError must reproduce; actual20kdraws/12family
bootstrap and malformed-value/key/length fixtures must pass the repaired audit.

This new execution repeats full saved arithmetic in its own directory, not any
model forward/backward/optimizer/training. Inherits all2057 prior digests plus
the first recovery failure and sources. Same conservative task memory1.3 policy.
One preflight, unique0trigger independent OS worker, one startup, one elapsed50%
then completion node. Source/science/resource/ID barrier stops with evidence;
no unlimited retry, no next experiment authorization, no old failure overwrite.

First new comparator fixture stopped before source lock/production: saved JSON
numeric lists correspond to independently computed NumPy arrays. Repaired type
handling accepts arrays as the sequence counterpart with exact length and same
scalar tolerance. Failed code/trace preserved automatically in this directory's
failures. No numerical value or tolerance was changed to pass a scientific gate.
Second preflight stopped on exact configuration-object comparison: this child
adds peak provenance to MEMORY_ESTIMATE. Same numerical estimate, but the cache
must read this child's full JSON after path redirection. Mismatch/code/trace are
preserved. A documentation patch was rejected before any file edit; confirmed
both files unchanged before applying the corrected hunk. No scientific change.
