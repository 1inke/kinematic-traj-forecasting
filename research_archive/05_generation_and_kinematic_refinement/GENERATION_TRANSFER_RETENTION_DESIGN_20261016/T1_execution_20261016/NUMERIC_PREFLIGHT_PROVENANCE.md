# First preflight corrections before source lock or production launch

First failure: constant-.2 synthetic boundary had NumPy mean .20000000000000004 versus CUDA .2, so one interpreted J>1 and the other J=1. Scientific thresholds/tolerance remain unchanged; the exact-boundary fixture now uses endpoint .4 with ADE strictly below .2. This is an analytically exact J=1 fixture rather than an evaluation epsilon.

A separate explicit mixed-success test exposed incorrect GT subsetting: the new keep implementation selected pred[valid] but not gt[valid], creating unintended batch broadcasting. Corrected both to the identical valid subset as the preregistered pseudocode. No old source edited; first failed source/trace are preserved under failures. Complete preflight must rerun; no production worker/model has started. This correction is not evidence of scientific benefit.

Second full preflight reached actual GPU8 A successfully, then the legacy comparator could not traverse the new list-of-dictionaries strata schema. Added a new recursive dict/list/tuple comparator (same2e-12 numeric tolerance, exact structural and bool/string checks). Actual generated values were not reported mismatched. Second failed sources and trace remain archived; full preflight must pass before launch.
