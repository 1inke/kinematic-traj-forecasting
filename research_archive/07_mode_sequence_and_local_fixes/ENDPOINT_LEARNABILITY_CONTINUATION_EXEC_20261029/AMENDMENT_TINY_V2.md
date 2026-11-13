# END_TINY_SAMPLING_AMENDMENT_V2_20261029

The human explicitly authorized the continuation design on 2026-10-29.
Keep the old hash tag endpoint-temporal-tiny-v1|sample_id, ascending SHA256 hex
then integer sample_id. The first64 are active train windows. The second64 are
the first remaining train windows in that same ordering, whether active or not.
No replacement, no held/val borrowing, no outcome-dependent resampling.

Keep H seed42, 4770 parameters, all20 candidates and unchanged targets, batch32,
AdamW .001/wd.0001/clip5, 256 updates, readouts0/256, and active-candidate MSE
relative drop>=90%. Old V1 STOP/source/results stay unchanged. This is not
positive/negative balancing and not a replication of the old two strata.
