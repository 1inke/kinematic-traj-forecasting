# Current decisions and remaining admissions

Human 2026-11-06: configure map/DSF interfaces. Later clarification: DSF can generalize to other latent-to-trajectory generators; first try the current generator, pretrain a CVAE only if that fails.

## Selected route

Use the existing trained F16+h3 frozen epoch8 EMA and expose deterministic per-slot query control. The actual FIT8 GPU test demonstrates native identity, reachable path changes, DPP control gradient and immutable old parameters. The necessary engineering interface is feasible. **No CVAE pretraining is needed at this point**; no CVAE or scientific DSF training has been started.

This is not the paper's Gaussian latent model. Disable the Gaussian-radius arm for the current deterministic generator. Use explicit labels current-frozen/reference, matched uniform-DPP and quality-DPP when a formal training protocol is prepared. Map-dependent arms remain conditional. Do not silently call a new non-Gaussian radius constraint the original baseline, change existing losses/weights or report extra model parameters as budget compliant.

## Map assets and unresolved details

The earlier statement “no scene assets exist” is now obsolete: official Y-net semantic masks and H for all current recording aliases have been retrieved and SHA recorded (17 files; only 41752 bytes fetched via ZIP ranges). Known binary maps, shared masks across recordings and the all-one uni_examples map are explicitly retained.

However, asset presence alone cannot certify the pedestrian 0/1 class policy, exact static annotation provenance/permission, calibration and valid-domain evidence. Keep zero production-admitted maps until those are verified. Past-only overlay projects all 211776 observed points into the supplied image dimensions, but that is a consistency check, not independent metric calibration or a license/class certificate. Do not choose labels or H based on future/coverage results, and do not remove any recording.

## Three inline ARS counterevidence checkpoints (same assistant, not external)

DA1 REVISE resolved: a Gaussian CVAE was incorrectly treated as a universal DSF requirement. Paper4.1 permits other proper generators. Current frozen h3 is a deterministic control-latent adaptation; original Gaussian-radius interpretation remains inapplicable.

DA2 PASS engineering / unresolved scientific effects: actual checkpoint gradients and fixed perturbations prove connection, not useful new modes. Slot templates and scale bound the reachable path family, and zero-last initially blocks earlier sampler layers until its last layer changes. No optimizer steps were made; it would be false to claim training/resume or coverage improvement.

DA3 PASS limited delivery: code and real asset acquisition are delivered with missing scientific map evidence, parameter overhead and no formal training explicit. Unknown maps are never free/safe by default. Old Stage A science and all old files/checkpoints/results/PAUSED records remain unchanged.
