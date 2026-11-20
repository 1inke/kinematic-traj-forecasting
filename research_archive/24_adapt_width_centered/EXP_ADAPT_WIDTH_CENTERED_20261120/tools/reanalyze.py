"""Recompute selection and gate statistics from the archived aggregate tables."""
import gzip
from hashlib import sha256
from itertools import combinations
import json
from math import isclose
from pathlib import Path
from statistics import mean

ARCHIVE = Path(__file__).resolve().parents[1]
SNAPSHOT = ARCHIVE / "snapshot"
OBS = ("stationary", "steady", "other")
ISO = "isotropic_coarse_FDE"
ANISO = "anisotropic_vs_isotropic_coarse_FDE"
ORACLE = "oracle_future_FDE"
ES = "coarse_path_weighted_ES"
ENERGY = ("path_weighted", "path_equal", "endpoint_weighted", "endpoint_equal")


def load(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def close(a, b):
    assert isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def numbers(key):
    return tuple(float(s) for s in key.split(","))


def preference(key, score):
    values = numbers(key)
    return score, *(abs(round(v - 1, 10)) for v in values), *values


def reselect(curves, policies):
    prefix = {"obs": "", "bin": "bin_", "label": "future_"}
    count = 0
    for policy in policies.values():
        group = curves[policy["grid"]]
        metric = policy["metric"]
        for name, selected in policy["by_group"].items():
            row = group[prefix[policy["labels"]] + name]
            grid = row["scales"]
            best = min(grid, key=lambda key: preference(key, grid[key][metric]))
            chosen = selected["scale"]
            assert numbers(best) == (tuple(chosen) if isinstance(chosen, list) else (chosen,))
            assert row["population"]["windows"] == selected["n"]
            close(grid[best][metric], selected["value"])
            close(grid["1.0,1.0" if isinstance(chosen, list) else "1.0"][metric], selected["id_value"])
            count += 1
    return count


def audit_curves(curves):
    count = 0
    for split, tables in curves.items():
        for grid, groups in tables.items():
            size = 144 if grid == "aniso" else 13
            for row in groups.values():
                scales = row["scales"]
                assert len(scales) == size and row["population"]["windows"] > 0
                for metrics in scales.values():
                    for energy in ENERGY:
                        a, s, es = (metrics[f"{energy}_{part}"] for part in ("A", "S", "ES"))
                        if split == "dev":
                            a, s, es = (part["estimate"] for part in (a, s, es))
                        close(a-s, es)
                    count += 1
    return count


def gate(effect, group, threshold):
    row = effect["groups"]["all_dev"][group]
    without = effect["groups"]["without_zara02"][group]
    record = effect["recordings"][group]
    values = record["summed_gain_by_recording_m"]
    net_gain = -row["delta_minFDE"]["estimate"] * row["population"]["windows"]
    close(net_gain, record["summed_net_gain_m"])
    close(net_gain, sum(values.values()))
    max_share = max(values.values()) / net_gain if net_gain > 0 else None
    if max_share is not None:
        close(max_share, record["maximum_contribution_share"])
    margin = row["relative_FDE_gain"]
    passed = (margin >= threshold and row["delta_minFDE"]["ci95"][1] < 0 and
              without["relative_FDE_gain"] > 0 and max_share is not None and max_share <= .5)
    return passed, margin


def audit_means(effects):
    count = 0
    for comparison in effects.values():
        for subset in comparison["groups"].values():
            for row in subset.values():
                delta = row["delta_minFDE"]["estimate"]
                before = row["minFDE"]["estimate"] - delta
                close(-delta / before, row["relative_FDE_gain"])
                for energy in ENERGY:
                    close(row[f"{energy}_A"]["estimate"]-row[f"{energy}_S"]["estimate"],
                          row[f"{energy}_ES"]["estimate"])
                count += 1
    return count


def recompute(run, selections, effects, descriptors, curves, recorded):
    policies = selections["policies"]
    selected = reselect(curves[selections["selection_split"]], policies)
    checked = audit_curves(curves)
    paired = audit_means(effects)
    c1 = {g: gate(effects[ISO], g, threshold) for g, threshold in (("overall", .02), ("hard", .03))}
    c2, _ = gate(effects[ANISO], "overall", .01)
    c3 = effects[ORACLE]["groups"]["all_dev"]["overall"]["relative_FDE_gain"] < .02
    prefs = {g: policies[ES]["by_group"][g]["scale"] for g in OBS}
    c4 = round(max(prefs.values())-min(prefs.values()), 10) >= .2
    harm = {group: -effects[ES]["groups"]["all_dev"][group]["relative_FDE_gain"]
            for group in ("overall", "hard")}
    flat = {}
    for g in ("steady", "other"):
        row = curves["dev"]["iso"][g]["scales"]
        for scale, ratio in descriptors["flatness"][g].items():
            close(row[scale]["minFDE"]["estimate"] / row["1.0"]["minFDE"]["estimate"], ratio)
        flat[g] = all(abs(descriptors["flatness"][g][scale]-1) <= .02 for scale in ("0.8", "1.2"))
    flags = dict(C1=any(x[0] for x in c1.values()), C2=c2, C3=c3, C4=c4, C6=all(flat.values()))
    for flag, val in flags.items():
        assert val == recorded[flag]["pass_flag"], (run, flag)
    assert prefs == recorded["C4"]["scale"]
    return dict(flags=flags, gains={g: x[1] for g, x in c1.items()}, prefs=prefs, harm=harm,
                checked_curves=checked, selections=selected, paired_groups=paired)


def headline(effects, keys):
    def relative(case, group, scope="all_dev"):
        return mean(effects[k][case]["groups"][scope][group]["relative_FDE_gain"] for k in keys)

    def weighted_es(scope):
        group = [effects[k][ES]["groups"][scope]["overall"] for k in keys]
        delta = mean(row["delta_path_weighted_ES"]["estimate"] for row in group)
        return -delta / (mean(row["path_weighted_ES"]["estimate"] for row in group) - delta)

    out = {case: dict(overall=relative(case, "overall"), hard=relative(case, "hard"),
                      overall_without_zara02=relative(case, "overall", "without_zara02"))
           for case in (ISO, ANISO, ORACLE, ES, "coarse_path_equal_ES")}
    out[ES]["weighted_ES_gain"] = weighted_es("all_dev")
    out[ES]["weighted_ES_gain_without_zara02"] = weighted_es("without_zara02")
    return out


def main():
    result, selections, descriptors = (load(SNAPSHOT / f"{name}.json")
                                      for name in ("RESULT", "selections", "descriptors"))
    effects_path = SNAPSHOT / "dev_effects.json.gz"
    with gzip.open(effects_path, "rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    assert digest.hexdigest() == load(SNAPSHOT / "DONE.json")["files"]["dev_effects.json"]
    effects = load(effects_path)
    assert result["by_arm_seed"].keys() == selections.keys() == descriptors.keys() == effects.keys()
    per_run = {}
    for key in selections:
        per_run[key] = recompute(key, selections[key], effects[key], descriptors[key],
                                load(SNAPSHOT / f"curves_{key}.json"), result["by_arm_seed"][key])
    keys = [k for k in per_run if k.startswith("B0_")]
    assert len(keys) == 5 and len(per_run) == 7
    counts = {flag: sum(per_run[k]["flags"][flag] for k in keys) for flag in per_run[keys[0]]["flags"]}
    flags = {flag: n >= 4 for flag, n in counts.items()}
    for flag, n in counts.items():
        assert n == result["summary"][flag]["passing_B0"]
    confirm = {g: True for g in ("overall", "hard")}
    for seed in (42, 43):
        b0, held = (per_run[f"{arm}_s{seed}"] for arm in ("B0", "E1B"))
        for g in confirm:
            ref, observed = b0["gains"][g], held["gains"][g]
            confirm[g] &= ref > 0 and observed > 0 and observed >= .5*ref
        pairs = list(combinations(OBS, 2))
        difference = lambda pair, r: round(r["prefs"][pair[1]]-r["prefs"][pair[0]], 10)
        strongest = max(pairs, key=lambda pair: abs(difference(pair, b0)))
        flags["C4"] &= difference(strongest, b0)*difference(strongest, held) > 0
    flags["C1H"] = any(confirm.values())
    flags["C5"] = all(mean(per_run[k]["harm"][g] for k in keys) <= cap
                      for g, cap in (("overall", .01), ("hard", .02)))
    assert flags == {k: v["pass_flag"] for k, v in result["summary"].items()}
    print(json.dumps(dict(status="PASS", runs=len(per_run), markers=flags,
                          **{key: sum(x[key] for x in per_run.values())
                             for key in ("checked_curves", "selections", "paired_groups")},
                          results=headline(effects, keys)), indent=2))


if __name__ == "__main__":
    main()

