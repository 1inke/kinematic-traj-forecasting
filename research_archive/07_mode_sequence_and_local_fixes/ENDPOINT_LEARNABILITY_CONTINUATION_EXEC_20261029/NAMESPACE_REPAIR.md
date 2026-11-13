# Explicit namespace-only repair, before any tiny model update

First tiny invocation failed while importing old learning_preflight.py:
its sibling import p0_driver was not on the new execution's sys.path. The stack
trace terminates before helper construction returned, before Engine creation.
The failed locked driver, source manifest and FAILURE receipt are unchanged.

The new adapter appends only the original execution ROOT for that sibling import
and invokes the unchanged locked run_tiny function. All cached/core functions are
byte-identical, all selection128 IDs and optimizer settings are unchanged.
There was no completed/partial tiny checkpoint to repeat; the adapter refuses to
start if an initial tiny checkpoint already exists. Its source/failure receipts
are bound by namespace_adapter_source_extension.json before the new invocation.
