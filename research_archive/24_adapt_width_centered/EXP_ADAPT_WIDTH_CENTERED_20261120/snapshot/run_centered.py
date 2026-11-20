"""Independent CPU runner for the prespecified centered width experiment."""
import argparse
from pathlib import Path
import time
import traceback

from centered_inputs import (ISO, OBS, ROOT, RUNS, dev_groups, draws_for, init, load_pair,
                             read, sha, status, verify, widths, write)
from centered_analysis import curve_tables, descriptive, dev_effects, eq_grid, settings, use
from centered_decisions import ANISO_V_ISO, ISO as POLICY_ISO, evaluate, report
from centered_metrics import all_grids


def compute_grids(d, paths, weights, h, draws, notify):
    iso, aniso = all_grids(paths, d["target_rel"], weights, d["rotation"][:, :, 0], h, draws, notify)
    notify("EQUAL_CENTER", 0, len(ISO))
    equal = eq_grid(paths, d, weights, h, draws, ISO, iso[1.])
    return iso, aniso, equal


def run_one(arm, seed, ordinal, data, archive, cuts, files, sample=None):
    select, paired = load_pair(arm, seed, data, files)
    results, masks = {}, {}
    clock = time.monotonic()

    def notify(component, step=0, total=1):
        status("RUNNING", arm=arm, seed=seed, arms_complete=ordinal-1, arms_total=len(RUNS),
               split=split, component=component, step=step, component_total=total,
               elapsed_in_arm_seconds=round(time.monotonic()-clock, 1))

    for split in (select, "dev"):
        d = data[split]
        x, w = paired[split]
        h = widths(archive, seed, d)
        if sample:
            d = {k: v[:sample] for k, v in d.items()}
            x, w, h = x[:sample], w[:sample], h[:sample]
        masks[split] = dev_groups(d, x, w)
        notify("DRAWING")
        results[split] = compute_grids(d, x, w, h, draws_for(len(x)), notify)
    fit, dev = data[select], data["dev"]
    if sample:
        fit, dev = ({k: v[:sample] for k, v in d.items()} for d in (fit, dev))
    selected_iso, selected_aniso, _ = results[select]
    policies = settings(*results[select], fit)
    iso, aniso, equal = results["dev"]
    grids = dict(iso=iso, aniso=aniso, equal_center=equal)
    effects = {name: use(grids[spec["grid"]], dev[spec["labels"]], spec["by_group"])
               for name, spec in policies.items()}
    comparisons = {name: (iso[1.], field) for name, field in effects.items()}
    comparisons[ANISO_V_ISO] = effects[POLICY_ISO], effects["anisotropic_coarse_FDE"]
    notify("DEV_PAIRED_EFFECTS")
    measured = dev_effects(comparisons, masks["dev"], dev)
    report_desc = descriptive(iso, aniso, dev, masks["dev"], fit, masks[select], selected_iso, selected_aniso)
    curves = {}
    for split in (select, "dev"):
        d = fit if split == select else dev
        group = masks[split]
        coarse = {g: group[g] for g in OBS}
        curves[split] = {}
        datasets = zip(("iso", "aniso", "equal_center"), results[split], (group, coarse, coarse))
        for name, grid, subset in datasets:
            notify(f"SUMMARIZE_{name}")
            curves[split][name] = curve_tables(grid, subset, d["track"], split == "dev")
    return (dict(selection_split=select, speed_cuts_mps=cuts[select], policies=policies),
            measured, report_desc, curves)


def source_hashes():
    names = sorted([*ROOT.glob("*.py"), *ROOT.glob("standalone/*.ps1"), ROOT/"PROTOCOL.md", ROOT/"DESIGN.md"])
    return {str(p.relative_to(ROOT)): sha(p) for p in names}


def start(sample):
    seal = ROOT/"DONE.json"
    if sample is None and seal.exists():
        assert all(sha(ROOT/name) == digest for name, digest in read(seal)["files"].items())
        return
    status("VALIDATE_INPUTS")
    fixtures = init()
    data, archive, cuts, replay, files = fixtures
    pre = read(ROOT/"PREFLIGHT.json")
    assert pre["status"] == "PASS" and len(pre["PF1"]) == 5 and len(pre["PF2_PF3_PF4"]) == 7
    assert all(max(p["max_coordinate_m"], p["max_logit"]) <= 1e-4 for p in replay["checks"].values())
    for src, digest in pre["files"].items():
        verify(Path(src), digest, files)
    if sample is not None:
        run_one("B0", 42, 1, data, archive, cuts, files, sample)
        status("SMOKE_PASSED", windows=sample)
        return
    started = time.monotonic()
    sources = source_hashes()
    signature = dict(sources=sources, audited_inputs=files.copy(), preflight=sha(ROOT/"PREFLIGHT.json"))
    staging = ROOT/"checkpoints"
    staging.mkdir(exist_ok=True)
    selections, effects, descs = {}, {}, {}
    outputs = ["PREFLIGHT.json", "HELDOUT_REPLAY.json"]
    for ordinal, (arm, seed) in enumerate(RUNS, start=1):
        key = f"{arm}_s{seed}"
        curve_name = f"curves_{key}.json"
        p = staging/f"{key}.json"
        curves = ROOT/curve_name
        cached = read(p) if p.exists() and curves.exists() else {}
        if cached and cached["signature"] == signature and cached["curve_sha256"] == sha(curves):
            selection, dev, descriptors = (cached[n] for n in ("selection", "dev", "descriptors"))
        else:
            selection, dev, descriptors, grid = run_one(arm, seed, ordinal, data, archive, cuts, files)
            write(curves, grid)
            del grid
            write(p, dict(selection=selection, dev=dev, descriptors=descriptors,
                          signature=signature, curve_sha256=sha(curves)))
        selections[key], effects[key], descs[key] = selection, dev, descriptors
        outputs.append(curve_name)
        for label, block in (("selections", selections), ("dev_effects", effects), ("descriptors", descs)):
            write(ROOT/f"{label}.json", block)
        print(f"Sealed {key} ({ordinal}/{len(RUNS)})", flush=True)
    result = evaluate(selections, effects, descs)
    write(ROOT/"RESULT.json", result)
    (ROOT/"REPORT.md").write_text(report(result), encoding="utf-8")
    outputs += ["selections.json", "dev_effects.json", "descriptors.json", "RESULT.json", "REPORT.md"]
    write(seal, dict(status="PASS", files={name: sha(ROOT/name) for name in outputs},
                     sources=sources, elapsed_seconds=round(time.monotonic()-started, 1)))
    status("DONE", arms_complete=len(RUNS), arms_total=len(RUNS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int)
    args = parser.parse_args()
    try:
        start(args.smoke)
    except Exception:
        status("FAILED", traceback=traceback.format_exc())
        raise

