# FAR_GOAL H0/H1/H2 execution, authorized 2026-10-19

The six parent Markdown files are the unchanged preregistered design. Human's
latest instruction authorizes execution, overriding their DESIGN_ONLY wording.
H0 is implemented first; H1 is NOT launched until actual H0 and engineering
preflight pass. No old worker or paused monitor is resumed.

H0 reads immutable inputs and target-label world coordinates, independently
reproduces the old float32 local rotation (direction from observed points 5→8,
slow threshold .03), and conservatively masks ambiguous start times. It does not
regenerate neighbors or remove missing-long-label windows. Existing full label
files, or original eth train/val shards when no full file exists, are merged
only by exact sequence/pedestrian/source-index identity. Conflicts are recorded;
no nearest-coordinate matching, interpolation, extrapolation, or new downloads.

Resource H0 planning: E=2GiB, comprising Python/NumPy+compressed input loading
under 1GiB, raw mmap working set under .5GiB and registry/labels/hash scratch
under .5GiB. Actual retained peak and host physical/commit remaining are guarded
at factor1.3. This is not an old mechanical 6GiB threshold; H1 needs its own
measured estimate and preflight.

ARS methodology and counterargument review is inline by the same assistant,
not independent external review. DA1: long-label selection and future-dependent
neighbor selection remain risks; matched auxiliary masks, fixed ALL denominators,
and immutable old observed cache address the new comparison, not legacy bias.
