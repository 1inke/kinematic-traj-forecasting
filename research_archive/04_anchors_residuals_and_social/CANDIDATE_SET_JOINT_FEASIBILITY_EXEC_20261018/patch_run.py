from pathlib import Path
p = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\run_k0.py")
t = p.read_text(encoding="utf-8")
old = '''    if do_k0b and not k0a["quality_upper_bound_negation"]:
        # Still run B even if union fails some clauses? Design: if union quality can't reach, skip more combo search for that quality.
        # We still attempt if only motion fails on union — quality UB negation is quality-only.
        pass
    if do_k0b:'''
new = '''    if do_k0b and k0a["quality_upper_bound_negation"]:
        do_k0b = False
        result["relaxation_certificate_status"] = "QUALITY_UPPER_BOUND_NEGATION"
        result["pool_joint_witness"] = False
        dump(out_split / "k0b_skipped_quality_ub.json", dict(
            reason="union40 quality clauses already fail W; K20 cannot improve quality beyond union",
            fail_clauses=k0a["union_quality_fail_clauses"],
        ))
    if do_k0b:'''
if old not in t:
    raise SystemExit("old block missing")
p.write_text(t.replace(old, new), encoding="utf-8")
print("patched ok")
