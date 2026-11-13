"""评测 native ∪ KIN 篮子覆盖与互补性。优先复用 SPEED_CLAMP 的 KIN checkpoint。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
CLAMP = ROOT / "KINEMATIC_DECODER_SPEED_CLAMP_20261018"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLAMP))

from kdec.data import load_bundle
from kdec.metrics import errors, motion_stats, motion_gate, evaluate_split, masks
from kdec import models as M

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
KIN_CEILING = 0.5646


@torch.no_grad()
def predict_kin(net, past, batch=256):
    net.eval()
    rows = []
    for i in range(0, len(past), batch):
        p = torch.from_numpy(past[i:i+batch].astype(np.float32)).to(DEVICE)
        rows.append(net(p)["paths"].detach().cpu().numpy())
    return np.concatenate(rows, 0)


def union_metrics(native, kin_paths, gt, groups):
    """40 候选 = native(20)+kin(20)。"""
    union = np.concatenate([native, kin_paths], axis=1)
    # standard evaluate on union vs native reference
    base = evaluate_split(kin_paths, gt, native, groups)
    uni = evaluate_split(union, gt, native, groups)
    # complementarity detail on overall
    _, _, _, n_good, _, _ = errors(native, gt)
    _, _, _, k_good, _, _ = errors(kin_paths, gt)
    _, _, _, u_good, _, _ = errors(union, gt)
    n_cov = n_good.any(1)
    k_cov = k_good.any(1)
    u_cov = u_good.any(1)
    only_n = int((n_cov & ~k_cov).sum())
    only_k = int((k_cov & ~n_cov).sum())
    both = int((n_cov & k_cov).sum())
    neither = int((~n_cov & ~k_cov).sum())
    # redundancy: fraction of KIN-good samples already native-good
    kin_good_n = int(k_cov.sum())
    redun = float(both / kin_good_n) if kin_good_n else 0.0
    return dict(
        KIN=base,
        UNION=uni,
        complementarity=dict(
            N=int(len(gt)),
            native_only=only_n,
            kin_only=only_k,
            both=both,
            neither=neither,
            kin_redundancy_among_kin_covered=redun,
            union_coverage=float(u_cov.mean()),
            native_coverage=float(n_cov.mean()),
            kin_coverage=float(k_cov.mean()),
            lift_vs_native_pp=float((u_cov.mean() - n_cov.mean()) * 100),
            lift_vs_kin_pp=float((u_cov.mean() - k_cov.mean()) * 100),
        ),
    )


def decide(comp, kin_outer):
    gate = kin_outer["_motion"]["gate"]["status"]
    u = comp["union_coverage"]
    n = comp["native_coverage"]
    k = comp["kin_coverage"]
    return dict(
        union_beats_native=bool(u + 1e-12 > n),
        union_beats_either=bool(u + 1e-12 > max(n, k)),
        lift_vs_native_pp=comp["lift_vs_native_pp"],
        kin_only_rescues=comp["kin_only"],
        kin_motion=gate,
        # 强门：并集相对原生有可观增益（≥1pp）且 KIN 运动 PASS
        strong_complement=bool(gate == "PASS" and comp["lift_vs_native_pp"] >= 1.0),
        # 弱门：有任意 kin_only rescue
        weak_complement=bool(comp["kin_only"] > 0 and gate == "PASS"),
        union_coverage=u,
        native_coverage=n,
        kin_coverage=k,
        beats_kin_ceiling_as_union=bool(u >= KIN_CEILING),
        beats_native_as_union=bool(u + 1e-12 >= n),
    )


def load_kin_net(ckpt_path, v_max, a_max):
    net = M.build("KIN", v_max=v_max, a_max=a_max).to(DEVICE)
    blob = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(blob["state_dict"])
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--split", default="outer", choices=["outer", "FIT", "inner"])
    ap.add_argument("--retrain-smoke", action="store_true",
                    help="若无 ckpt 则短训 smoke；默认只评现成 SPEED_CLAMP KIN")
    args = ap.parse_args()

    data = load_bundle()
    fit_m = motion_stats(data["native"][data["split"]["FIT"]])
    v_max = fit_m["speed_p95"] * 0.995
    a_max = fit_m["accel_p95"] * 0.90

    out_dir = HERE / "results" / "union"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for seed in args.seeds:
        ckpt = CLAMP / "results" / "short" / f"KIN_seed{seed}.pt"
        if not ckpt.exists():
            print("MISSING", ckpt)
            if args.retrain_smoke:
                from kdec.train import train_arm
                print("retrain smoke KIN seed", seed)
                train_arm("KIN", data, seed=seed, epochs=2, batch=128, subset=2048,
                          v_max=v_max, a_max=a_max, motion_w=0.05,
                          motion_refs=dict(speed_p95=fit_m["speed_p95"], accel_mean=fit_m["accel_mean"]),
                          out_dir=HERE / "results" / "smoke")
                ckpt = HERE / "results" / "smoke" / f"KIN_seed{seed}.pt"
            else:
                continue
        net = load_kin_net(ckpt, v_max, a_max)
        ix = data["split"][args.split]
        past = data["past"][ix]
        native = data["native"][ix]
        gt = data["gt"][ix]
        groups = data["groups"][ix]
        kin_paths = predict_kin(net, past)
        packed = union_metrics(native, kin_paths, gt, groups)
        decision = decide(packed["complementarity"], packed["KIN"])
        row = dict(seed=seed, split=args.split, ckpt=str(ckpt),
                   complementarity=packed["complementarity"],
                   KIN_overall=packed["KIN"]["overall"],
                   UNION_overall=packed["UNION"]["overall"],
                   KIN_motion=packed["KIN"]["_motion"],
                   decision=decision)
        (out_dir / f"UNION_seed{seed}_{args.split}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        all_rows.append(row)
        print(json.dumps({"seed": seed, "decision": decision}, ensure_ascii=False), flush=True)

    if not all_rows:
        raise SystemExit("no seeds evaluated")

    lifts = [r["decision"]["lift_vs_native_pp"] for r in all_rows]
    summary = dict(
        n_seeds=len(all_rows),
        split=args.split,
        union_coverage_mean=float(np.mean([r["decision"]["union_coverage"] for r in all_rows])),
        native_coverage=all_rows[0]["decision"]["native_coverage"],
        kin_coverage_mean=float(np.mean([r["decision"]["kin_coverage"] for r in all_rows])),
        lift_vs_native_pp_mean=float(np.mean(lifts)),
        lift_vs_native_pp_std=float(np.std(lifts)),
        kin_only_mean=float(np.mean([r["complementarity"]["kin_only"] for r in all_rows])),
        redundancy_mean=float(np.mean([r["complementarity"]["kin_redundancy_among_kin_covered"] for r in all_rows])),
        strong_complement_count=sum(1 for r in all_rows if r["decision"]["strong_complement"]),
        weak_complement_count=sum(1 for r in all_rows if r["decision"]["weak_complement"]),
        kin_motion_pass=sum(1 for r in all_rows if r["decision"]["kin_motion"] == "PASS"),
        gate_HIT=bool(sum(1 for r in all_rows if r["decision"]["strong_complement"]) >= 2),
        rows=[{k: r[k] for k in ("seed", "decision", "complementarity")} for r in all_rows],
    )
    (HERE / "results" / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 结果 HYBRID_NATIVE_KIN_UNION",
        "",
        f"- split: {args.split}",
        f"- union cov mean: {summary['union_coverage_mean']:.4f}",
        f"- native cov: {summary['native_coverage']:.4f}",
        f"- kin cov mean: {summary['kin_coverage_mean']:.4f}",
        f"- lift vs native: {summary['lift_vs_native_pp_mean']:.2f}±{summary['lift_vs_native_pp_std']:.2f} pp",
        f"- kin_only rescues mean: {summary['kin_only_mean']:.1f}",
        f"- KIN redundancy (among KIN-covered): {summary['redundancy_mean']:.3f}",
        f"- strong_complement: {summary['strong_complement_count']}/{summary['n_seeds']}",
        f"- gate_HIT (≥2 strong): {summary['gate_HIT']}",
        "",
        "| seed | union | native | kin | lift_pp | kin_only | both | neither | kin_motion |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in all_rows:
        c = r["complementarity"]; d = r["decision"]
        lines.append(
            f"| {r['seed']} | {c['union_coverage']:.4f} | {c['native_coverage']:.4f} | {c['kin_coverage']:.4f} | {c['lift_vs_native_pp']:.2f} | {c['kin_only']} | {c['both']} | {c['neither']} | {d['kin_motion']} |"
        )
    text = "\n".join(lines) + "\n"
    (HERE / "results" / "RESULTS.md").write_text(text, encoding="utf-8")
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
