# Preflight adapter deepcopy repair, before any scientific run

Original preflight.py and PREFLIGHT_FAILURE.json plus all preflight/ fixtures are retained unchanged. Formal production has not started. Context cache is complete and source-SHA compatible; never re-encode it or restart Stage A.

Failure: `copy.deepcopy(adapter)` tries to pickle `adapter.api`, a referenced Python module, and raises TypeError cannot pickle module object. The adapter's model is an ordinary nn.Module and can be copied independently. New preflight_v2.py constructs a fresh wrapper from `deepcopy(adapter.model).cpu().double()` plus the same read-only API reference. Only the double-FD fixture construction and output/failure namespaces differ. Train/eval/objective/statistics/optimizer/data/source-contract code is unchanged. Both arms' full preflight is rerun in preflight_v2/; no completed scientific model retraining, because earlier work was throwaway preflight only.

The original PRE_SOURCE scientific file hashes remain valid. This file and preflight_v2.py are an explicit pre-production source extension, included in the production seal. No source/hash bypass, no scientific tolerance changes, no deletion.
