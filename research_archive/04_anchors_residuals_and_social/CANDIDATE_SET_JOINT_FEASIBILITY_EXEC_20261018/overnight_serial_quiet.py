# -*- coding: utf-8 -*-
"""Serial quiet overnight: finish K0 seed44 FIT_cal -> merge K0 analysis -> full K1."""
from __future__ import annotations
import json, os, sys, time, shutil, subprocess, ctypes
from datetime import datetime
from pathlib import Path

ROOT = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
LOG = ROOT / "OVERNIGHT_LOG.md"
STATUS = ROOT / "OVERNIGHT_STATUS.txt"
PY = ROOT / ".venv-k0" / "Scripts" / "python.exe"
K1_DIR = ROOT / "k1_exec"

def hkt():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")

def L(msg: str) -> None:
    line = f"- [{hkt()}] {msg}\n"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("# OVERNIGHT_LOG\n\n", encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)
    STATUS.write_text(
        f"overnight_policy: threads=2 Idle affinity=2cores one_heavy_at_a_time\n"
        f"last: {msg}\nupdated: {hkt()}\n",
        encoding="utf-8",
    )

def set_quiet_env():
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "2"
    os.environ["JOINT_FEAS_ROOT"] = r"<WORKSPACE_ROOT>"
    os.environ["JOINT_FEAS_EXEC"] = str(ROOT)
    os.environ["JOINT_FEAS_K1"] = str(K1_DIR)

def soften_process(pid: int) -> None:
    # Idle + affinity to first 2 logical processors
    try:
        import psutil  # type: ignore
        p = psutil.Process(pid)
        p.nice(psutil.IDLE_PRIORITY_CLASS)
        try:
            p.cpu_affinity([0, 1])
        except Exception:
            pass
    except Exception:
        try:
            subprocess.run(
                ["powershell","-NoProfile","-Command",
                 f"$p=Get-Process -Id {pid}; $p.PriorityClass='Idle'; try{{$p.ProcessorAffinity=3}}catch{{}}"],
                timeout=30, capture_output=True,
            )
        except Exception:
            pass

def run_one(cmd, log_path: Path, label: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    L(f"START {label}: {' '.join(str(c) for c in cmd)}")
    with log_path.open("ab") as out:
        out.write(f"\n==== {hkt()} START {label} ====\n".encode())
        out.flush()
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT)
        soften_process(p.pid)
        soften_process(os.getpid())
        rc = p.wait()
    L(f"END {label} rc={rc}")
    return rc

def backup_go():
    bak = ROOT / "logs" / "backup_before_k0_fitcal"
    bak.mkdir(parents=True, exist_ok=True)
    for name in ("GO_K1_REGISTER.json","RESULT_ANALYSIS.md","RESULT_ANALYSIS.json"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, bak / name)
    L(f"backed up GO_K1/RESULT_ANALYSIS -> {bak}")

def load_all_result_fields():
    results = []
    rdir = ROOT / "results"
    if not rdir.exists():
        return results
    for p in sorted(rdir.glob("seed*/**/result_fields.json")):
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            L(f"warn load {p}: {e}")
    return results

def merge_k0_analysis():
    """Rebuild appendix analysis from ALL result_fields without destroying GO_K1 truth."""
    all_results = load_all_result_fields()
    outer = [r for r in all_results if r.get("split") == "outer"]
    pool_ok = len(outer) >= 3 and all(r.get("pool_joint_witness") is True for r in outer)
    policy_ok = len(outer) >= 3 and all(r.get("policy_oracle_witness") is True for r in outer)
    go = dict(
        document="GO_K1_REGISTER",
        created_at=hkt(),
        timezone="Asia/Tokyo",
        exec_dir=str(ROOT),
        GO_K1=bool(pool_ok and policy_ok),
        reason="outer_3seed_pool_and_K0C_W" if (pool_ok and policy_ok) else "incomplete_or_fail",
        note="merged after seed44 FIT_cal; generator frozen",
        tag="ORACLE_DIAGNOSTIC_ONLY",
        author="tosaka_executor",
        secondary_splits={},
    )
    # secondary table
    lines = []
    lines.append("\n\n---\n\n## 附录 A · 副 split 补全（过夜续跑）\n\n")
    lines.append(f"- 写入时间：{hkt()}（Asia/Tokyo）\n")
    lines.append("- original_CAL：仍 **BLOCKED_NO_KIN_CACHE**（本夜不重开）\n\n")
    lines.append("| seed | split | pool_joint | policy_oracle | cert | change_good(k0c) | change ADE/FDE | W_pass(k0c) |\n")
    lines.append("|---:|---|---|---|---|---:|---|---|\n")
    for r in sorted(all_results, key=lambda x: (str(x.get("split")), int(x.get("seed", 0)))):
        ports = r.get("portfolios") or {}
        k0c = ports.get("k0c") or {}
        lines.append(
            f"| {r.get('seed')} | {r.get('split')} | {r.get('pool_joint_witness')} | "
            f"{r.get('policy_oracle_witness')} | {r.get('relaxation_certificate_status')} | "
            f"{k0c.get('change_good')} | {k0c.get('change_minADE')}/{k0c.get('change_minFDE')} | "
            f"{k0c.get('W_pass')} |\n"
        )
        go["secondary_splits"].setdefault(str(r.get("split")), {})[f"seed{r.get('seed')}"] = dict(
            pool=r.get("pool_joint_witness"),
            policy=r.get("policy_oracle_witness"),
            cert=r.get("relaxation_certificate_status"),
            k0c_W=k0c.get("W_pass"),
            change_good=k0c.get("change_good"),
        )
    # preserve prior narrative: append only
    md_path = ROOT / "RESULT_ANALYSIS.md"
    if md_path.exists():
        prev = md_path.read_text(encoding="utf-8")
        if "附录 A · 副 split 补全" in prev:
            # replace appendix section
            head = prev.split("## 附录 A · 副 split 补全")[0].rstrip()
            md_path.write_text(head + "\n" + "".join(lines), encoding="utf-8")
        else:
            md_path.write_text(prev.rstrip() + "\n" + "".join(lines), encoding="utf-8")
    else:
        md_path.write_text("# K0 RESULT_ANALYSIS\n" + "".join(lines), encoding="utf-8")
    # Do NOT clobber the rich GO_K1_REGISTER if already true — merge fields
    go_path = ROOT / "GO_K1_REGISTER.json"
    if go_path.exists():
        try:
            old = json.loads(go_path.read_text(encoding="utf-8"))
            if old.get("GO_K1") is True:
                old["secondary_splits_updated_at"] = hkt()
                old["secondary_splits"] = go["secondary_splits"]
                old["FIT_cal_seed44"] = "DONE" if (ROOT / "results/seed44/FIT_cal/result_fields.json").exists() else "MISSING"
                go_path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
                L("GO_K1 preserved=true; secondary_splits updated")
            else:
                go_path.write_text(json.dumps(go, ensure_ascii=False, indent=2), encoding="utf-8")
                L(f"GO_K1 rewritten GO_K1={go['GO_K1']}")
        except Exception as e:
            L(f"GO merge warn: {e}")
            go_path.write_text(json.dumps(go, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        go_path.write_text(json.dumps(go, ensure_ascii=False, indent=2), encoding="utf-8")
    # also dump merge snapshot without overwriting RESULT_ANALYSIS.json rich blob entirely
    snap = ROOT / "results" / "K0_SECONDARY_MERGE.json"
    snap.write_text(json.dumps(dict(updated=hkt(), n=len(all_results), go_k1=go.get("GO_K1"), secondary=go["secondary_splits"]), ensure_ascii=False, indent=2), encoding="utf-8")
    L(f"merged K0 secondary n_results={len(all_results)}")

def any_heavy_python() -> bool:
    try:
        r = subprocess.run(
            ["wmic","process","where","name='python.exe'","get","ProcessId,CommandLine","/FORMAT:LIST"],
            capture_output=True, text=True, timeout=60,
        )
        s = r.stdout or ""
        return ("run_k0.py" in s) or ("run_k1.py" in s) or ("overnight_serial" in s and "python" in s.lower())
    except Exception:
        return False

def main():
    set_quiet_env()
    soften_process(os.getpid())
    L("overnight_serial_quiet start")
    # wait if somehow another heavy job exists (not ourselves)
    # (we are the orchestrator; children are fine)
    rf = ROOT / "results" / "seed44" / "FIT_cal" / "result_fields.json"
    if not rf.exists():
        backup_go()
        # protect: after run_k0, it may overwrite GO with incomplete — we restore via merge
        rc = run_one(
            [str(PY), "-u", str(ROOT / "run_k0.py"), "--seeds", "44", "--splits", "FIT_cal"],
            ROOT / "logs" / "k0_seed44_FIT_cal_overnight.log",
            "K0_seed44_FIT_cal",
        )
        if rc != 0 or not rf.exists():
            L(f"K0 FIT_cal failed rc={rc} rf_exists={rf.exists()}; STOP before K1")
            return 3
        # restore GO from backup then merge
        bak = ROOT / "logs" / "backup_before_k0_fitcal" / "GO_K1_REGISTER.json"
        if bak.exists():
            shutil.copy2(bak, ROOT / "GO_K1_REGISTER.json")
            L("restored GO_K1_REGISTER from backup")
        merge_k0_analysis()
    else:
        L("FIT_cal seed44 already complete; merge only")
        merge_k0_analysis()

    # K1 formal
    K1_DIR.mkdir(parents=True, exist_ok=True)
    done = K1_DIR / "runtime_state.json"
    if done.exists():
        try:
            st = json.loads(done.read_text(encoding="utf-8"))
            if st.get("status") == "COMPLETE":
                L(f"K1 already COMPLETE GO_K2={st.get('GO_K2')}")
                return 0
        except Exception:
            pass
    marker = ROOT / "logs" / "k1_overnight.started"
    marker.write_text(hkt(), encoding="utf-8")
    rc = run_one(
        [str(PY), "-u", str(ROOT / "run_k1.py")],
        ROOT / "logs" / "k1_overnight.log",
        "K1_formal_I_vs_S",
    )
    if rc != 0:
        L(f"K1 failed rc={rc}")
        return rc
    go2 = K1_DIR / "GO_K2_REGISTER.json"
    if go2.exists():
        try:
            g = json.loads(go2.read_text(encoding="utf-8"))
            L(f"K1 done GO_K2={g.get('GO_K2')}")
            if g.get("GO_K2"):
                L("BREAKTHROUGH_CANDIDATE: GO_K2=true — parent should evaluate BREAKTHROUGH")
        except Exception as e:
            L(f"read GO_K2 err {e}")
    L("overnight_serial_quiet finished")
    return 0

if __name__ == "__main__":
    raise SystemExit(main() or 0)
