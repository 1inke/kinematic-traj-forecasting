# U/D experiment: actual contract-alignment check, not 8-epoch production

2026-11-02 human requested executing `<LOCAL_DESKTOP>/新调研任务/下一阶段U与D单因素实验设计文档_20261102.md`. Original design and all B0 science sources/models/results are preserved read-only. Current status: **STOP_CONTRACT_ALIGNMENT_WAIT_HUMAN**.

Prepared modules implement the supplied motion-state, GRU/Delta and ordinary/differential gated-attention equations. They default to the proposed96D output but were instantiated at the **actual B0 64D** interface solely for this alignment diagnostic. `decoder_adapter.py` is diagnostic glue to existing h3, not a claim that the conflicting96D architecture has been implemented as B0-equivalent.

## Evidence

`alignment_check/ACTUAL_GRADIENT_CHECK.json` binds the source, fixed hash8 actual FIT window IDs and all three updates' per-parameter gradient norms for each arm. All64 original B0 source digests and fresh initial weight/buffer equality verified. No old checkpoint warmstart: B0 `initial.pt` is read only to compare independently recreated fresh seed42 initialization. Initial paths/logits with zero residual are EXACT. Only12 diagnostic optimizer updates were performed, not four8epoch pilots, not a new formal result or method-effect test. No held effects measured, no OS worker/task/automation created, no completed PID/task/log queried, no deletion.

| Item | Supplied document | Actual locked B0 / measured diagnostic |
|---|---|---|
| context and mode feature |96D, `context_proj` / `anchor_head` names|64D, `context` / `mode_mlp` / `proposal_scale` / `proposal_delta`; do not silently widen old decoder|
| gradient clipping invariant |1.0|5.0 in protocol and actual trainer; changing to1 makes loss/optimizer intervention a second factor|
| full-model branch gate |Step1 `Wo.grad>0`, Step2 every internal tensor grad>0|All four arms Step1 `Wo.grad=0`, Step2 internal gate false, Step3 all internal tensors positive|
| full parameter count |approximately6–8k additions, below305k target|U-GRU301618 (+6144); U-Delta302514 (+7040); D-Gate307986 (+12512); D-DiffGate312082 (+16608)|

The existing B0 initializes `proposal_scale.weight`, `proposal_delta.weight` and final score layer to zero. At the first backward pass, the loss derivative with respect to context/mode features is zero. Hence even a nonzero state difference cannot give a nonzero gradient to the new zero output projection on that pass. After the original head opens, the new output projection opens next, and the internal branch opens on the third backward pass. This is a concrete chain-rule explanation and actual full-model GPU observation, not a claim that B0's unused legacy modules caused all prediction failures.

## Proposed narrow amendment — not yet adopted for production

1. Keep the actual B0 64D interface and clip5; use the same shared initial weights, fixed eight epoch batch-ID sequence, loss, scheduler, augmentation and EMA. Do not resize/refactor old layers or change clip merely to match mistaken prose.
2. Replace the impossible full-model two-step gate with a staged check: initial original path/logit EXACT; first original head gradients legal; second new `Wo` gradient positive; third every internal branch tensor positive. An isolated synthetic two-step branch test may additionally be run but cannot substitute for or be reported as the full-model gate.
3. Preserve exact added parameters and report D's305k target overage. Do not shrink attention width or alter the original backbone to force the number below the target. Human direction is needed if305k must instead be a hard admission cap.

This is an execution-contract issue, **not resource insufficiency and not a negative U/D scientific result**. There is no production ETA or startup claim yet. After amendment approval, the remaining full GPU/independent objective/FD, all-RNG two-epoch resume, full saved quality/cluster/motion, cost and budget preflight still must be executed before four independent eight-epoch pilots.

## ARS inline counter-review (same assistant, not external review)

- DA1: Literal dimensions, clipping and gradient gates conflict with the mandatory unchanged-B0 single-factor premise. Flag before production, retain measured evidence; do not invent a PASS.
- DA2: A first-step zero gradient is not permanent deadlock; actual third-step connectivity counters that inference. Parent diagnostic report's claims about unique roots/power/Hermite are not automatically certified by this check.
- DA3: No performance conclusions, seed expansion, V, combination or formal training are licensed by this diagnostic. The document's subjective “clear mechanism signal” seed-expansion language also needs a sealed objective rule before held results are used to select follow-ups. Source/ID, calibration and historical scientific gates remain unchanged.
