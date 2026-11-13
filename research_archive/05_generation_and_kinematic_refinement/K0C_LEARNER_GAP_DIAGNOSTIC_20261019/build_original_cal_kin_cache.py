# -*- coding: utf-8 -*-
"""Frozen KIN forward for original_CAL only. no_grad, CPU-first. No selector train."""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np

os.environ["KIN_FORCE_CPU"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

ROOT = Path(r"<WORKSPACE_ROOT>")
EXEC = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
SEL = ROOT / "UNION_SELECTOR_K40_TO_K20_20261018"
CLAMP = ROOT / "KINEMATIC_DECODER_SPEED_CLAMP_20261018"
CPV_ART = ROOT / "COMPLEMENTARY_POOL_VALIDATION_20261018" / "artifacts" / "v1"
OUT = ROOT / "K0C_LEARNER_GAP_DIAGNOSTIC_20261019"
CACHE = SEL / "results" / "cache_pred"
CACHE.mkdir(parents=True, exist_ok=True)
CPV_ART.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXEC))
from k0.data_io import load_bundle
from k0.metrics import motion_stats

SEEDS = (42, 43, 44)
status = dict(created_at=time.strftime("%Y-%m-%dT%H:%M:%S+09:00"), timezone="Asia/Tokyo",
              task="original_CAL_KIN_cache", force_cpu=True, seeds={})

def main():
    try:
        import torch
    except Exception as e:
        status["status"] = "BLOCKED"
        status["reason"] = f"torch_import_failed: {e}"
        (OUT / "CAL_CACHE_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print("BLOCKED torch", e)
        return

    data = load_bundle()
    ix = data["split"]["original_CAL"]
    assert len(ix) == 2879, len(ix)
    past_all = data["past"][ix].astype(np.float32)
    fit_m = motion_stats(data["native"][data["split"]["FIT"]])
    v_max = float(fit_m["speed_p95"] * 0.995)
    a_max = float(fit_m["accel_p95"] * 0.90)
    status["v_max"] = v_max
    status["a_max"] = a_max
    status["N"] = int(len(ix))

    sys.path.insert(0, str(SEL))
    from kdec import models as M

    device = torch.device("cpu")
    status["device"] = str(device)
    print(f"device={device} N={len(ix)} v_max={v_max:.4f} a_max={a_max:.4f}", flush=True)

    ok_all = True
    for seed in SEEDS:
        rec = dict(seed=seed)
        dest1 = CACHE / f"kin_original_CAL_seed{seed}.npy"
        dest2 = CPV_ART / f"kin_original_CAL_seed{seed}.npy"
        if dest1.exists() and dest2.exists():
            kin = np.load(dest1)
            rec.update(status="EXISTS", shape=list(kin.shape), path=str(dest1))
            status["seeds"][str(seed)] = rec
            print(f"seed{seed} EXISTS {kin.shape}", flush=True)
            continue
        ckpt = CLAMP / "results" / "short" / f"KIN_seed{seed}.pt"
        if not ckpt.exists():
            rec.update(status="BLOCKED", reason=f"missing_ckpt {ckpt}")
            status["seeds"][str(seed)] = rec
            ok_all = False
            print(f"seed{seed} BLOCKED no ckpt", flush=True)
            continue
        try:
            t0 = time.time()
            net = M.build("KIN", v_max=v_max, a_max=a_max).to(device)
            blob = torch.load(ckpt, map_location=device, weights_only=False)
            net.load_state_dict(blob["state_dict"])
            net.eval()
            rows = []
            bs = 64
            with torch.no_grad():
                for i in range(0, len(past_all), bs):
                    p = torch.from_numpy(past_all[i:i+bs]).to(device)
                    rows.append(net(p)["paths"].detach().cpu().numpy())
                    if i % (bs * 8) == 0:
                        print(f"  seed{seed} forward {i}/{len(past_all)}", flush=True)
            kin = np.concatenate(rows, 0).astype(np.float64)
            assert kin.shape == (2879, 20, 12, 2), kin.shape
            np.save(dest1, kin)
            np.save(dest2, kin)
            rec.update(
                status="OK", shape=list(kin.shape), dtype=str(kin.dtype),
                path_cache=str(dest1), path_art=str(dest2),
                wall_s=time.time()-t0,
            )
            status["seeds"][str(seed)] = rec
            print(f"seed{seed} OK {kin.shape} {rec['wall_s']:.1f}s", flush=True)
        except Exception as e:
            ok_all = False
            rec.update(status="BLOCKED", reason=str(e), traceback=traceback.format_exc()[-1500:])
            status["seeds"][str(seed)] = rec
            print(f"seed{seed} BLOCKED {e}", flush=True)

    status["status"] = "OK" if ok_all and all(status["seeds"][str(s)].get("status") in ("OK","EXISTS") for s in SEEDS) else "BLOCKED_OR_PARTIAL"
    (OUT / "CAL_CACHE_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    # append overnight log
    logp = EXEC / "OVERNIGHT_LOG.md"
    with logp.open("a", encoding="utf-8") as f:
        f.write(f"\n- {status['created_at']}: original_CAL KIN cache attempt status={status['status']}; see K0C_LEARNER_GAP_DIAGNOSTIC_20261019/CAL_CACHE_STATUS.json\n")
    print("FINAL", status["status"], flush=True)

if __name__ == "__main__":
    main()