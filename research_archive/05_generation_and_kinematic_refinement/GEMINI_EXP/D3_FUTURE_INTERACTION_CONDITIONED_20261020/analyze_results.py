import json
from pathlib import Path
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
r = json.load(open(EXP_DIR / 'RESULT.json', encoding='utf-8'))

print("=== D3: FUTURE INTERACTION CONDITIONED GENERATION ANALYSIS ===\n")

print("1. PRIMARY EVALUATION POPULATION: outer change (253 windows, 68 tracks)")
print("-" * 90)
header = f"{'Run':10s} | {'Cov(%)':7s} | {'Good':5s} | {'Rescue':6s} | {'Lost':5s} | {'Retain':6s} | {'J':6s} | {'ADE':6s} | {'MotPass':7s}"
print(header)
print("-" * 90)

for rk in sorted(r['runs'].keys()):
    m = r['runs'][rk]['outer']['change']
    print(f"{rk:10s} | {m['coverage']*100:6.2f}% | {m['good']:5d} | {m['rescue']:6d} | {m['lost']:5d} | {m['retained']:6d} | {m['J']:6.4f} | {m['ADE']:6.4f} | {str(m['motion_pass']):7s}")

print("\n2. DETAILED KINEMATICS AUDIT (outer change)")
print("-" * 90)
for rk in sorted(r['runs'].keys()):
    m = r['runs'][rk]['outer']['change']
    mot = m['motion']
    base = m['native_motion']
    print(f"[{rk}]")
    print(f"  Accel Mean: {mot['accel_mean']:.4f} (base: {base['accel_mean']:.4f}, ratio: {mot['accel_mean']/base['accel_mean']:.3f}x)")
    print(f"  Accel p95:  {mot['accel_p95']:.4f} (base: {base['accel_p95']:.4f}, ratio: {mot['accel_p95']/base['accel_p95']:.3f}x)")
    print(f"  Jerk Mean:  {mot['jerk_mean']:.4f} (base: {base['jerk_mean']:.4f}, ratio: {mot['jerk_mean']/base['jerk_mean']:.3f}x)")
    print(f"  Jerk p95:   {mot['jerk_p95']:.4f} (base: {base['jerk_p95']:.4f}, ratio: {mot['jerk_p95']/base['jerk_p95']:.3f}x)")
    print(f"  Speed p95:  {mot['speed_p95']:.4f} (base: {base['speed_p95']:.4f}, ratio: {mot['speed_p95']/base['speed_p95']:.3f}x)")

print("\n3. CLUSTERED PAIRED BOOTSTRAP TEST (Treatment B vs Control A, 20,000 draws)")
print("-" * 90)
print(f"{'Metric':12s} | {'Diff Mean':10s} | {'95% CI Lower':13s} | {'95% CI Upper':13s} | {'p-value':8s} | {'Crosses 0':10s}")
print("-" * 90)
for k, v in r['bootstrap_summary'].items():
    print(f"{k:12s} | {v['observed_mean_diff']:+10.4f} | {v['ci_lower']:+13.4f} | {v['ci_upper']:+13.4f} | {v['p_value']:8.4f} | {str(v['crosses_zero']):10s}")

print("\n4. CROSS-SPLIT GENERALIZATION AUDIT (Coverage % across FIT, outer, inner, CAL)")
print("-" * 90)
print(f"{'Run':10s} | {'FIT (overall)':14s} | {'outer (overall)':16s} | {'inner (overall)':16s} | {'CAL (overall)':14s}")
print("-" * 90)
for rk in sorted(r['runs'].keys()):
    runs = r['runs'][rk]
    fit_cov = runs['FIT']['overall']['coverage'] * 100
    out_cov = runs['outer']['overall']['coverage'] * 100
    inn_cov = runs['inner']['overall']['coverage'] * 100
    cal_cov = runs['original_CAL']['overall']['coverage'] * 100
    print(f"{rk:10s} | {fit_cov:13.2f}% | {out_cov:15.2f}% | {inn_cov:15.2f}% | {cal_cov:13.2f}%")
