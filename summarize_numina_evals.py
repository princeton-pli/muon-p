#!/usr/bin/env python
"""Collect pass@k eval summaries for the 12 numina SmolLM-135M seed runs."""
import json, os, re
from collections import defaultdict

RUNS = [
    "numina_SmolLM-135M_newtonschulz5_exp1-3_lr1e-4_newtonschulz5_lr1e-4_seed5836_6ksteps",
    "numina_SmolLM-135M_newtonschulz5_exp1-3_lr1e-4_newtonschulz5_lr1e-4_seed381_6ksteps",
    "numina_SmolLM-135M_newtonschulz5_exp1-3_lr1e-4_newtonschulz5_lr1e-4_seed823_6ksteps",
    "numina_SmolLM-135M_newtonschulz5_exp1-3_lr1e-4_newtonschulz5_lr1e-4_seed415_6ksteps",
    "numina_SmolLM-135M_adamw_exp1-3_lr1e-4_adamw_lr1e-4_seed381_6ksteps",
    "numina_SmolLM-135M_adamw_exp1-3_lr1e-4_adamw_lr1e-4_seed823_6ksteps",
    "numina_SmolLM-135M_halfpower_exp1-3_lr1e-4_halfpower_exp1-3_lr1e-4_seed823_6ksteps",
    "numina_SmolLM-135M_adamw_exp1-3_lr1e-4_adamw_lr1e-4_seed5836_6ksteps",
    "numina_SmolLM-135M_adamw_exp1-3_lr1e-4_adamw_lr1e-4_seed415_6ksteps",
    "numina_SmolLM-135M_halfpower_exp1-3_lr1e-4_halfpower_exp1-3_lr1e-4_seed381_6ksteps",
    "numina_SmolLM-135M_halfpower_exp1-3_lr1e-4_halfpower_exp1-3_lr1e-4_seed5836_6ksteps",
    "numina_SmolLM-135M_halfpower_exp1-3_lr1e-4_halfpower_exp1-3_lr1e-4_seed415_6ksteps",
]
RESULT_FILE = "pass_at_k_AI-MO_NuminaMath-CoT_n32_t0.8.json"

def optimizer(run):
    if "newtonschulz5" in run: return "newtonschulz5"
    if "adamw" in run: return "adamw"
    if "halfpower" in run: return "halfpower"
    return "?"

def seed(run):
    m = re.search(r"seed(\d+)", run)
    return m.group(1) if m else "?"

rows, missing = [], []
for r in RUNS:
    p = os.path.join("outputs", r, RESULT_FILE)
    if not os.path.exists(p):
        missing.append(r); continue
    with open(p) as f:
        d = json.load(f)
    rows.append((optimizer(r), seed(r), d))

# per-run table
kvals = sorted({int(k.split("@")[1]) for _,_,d in rows for k in d if k.startswith("pass@")})
hdr = ["optimizer", "seed"] + [f"pass@{k}" for k in kvals]
print("\n=== Per-run results ===")
print("  ".join(f"{h:>14}" for h in hdr))
for opt, sd, d in sorted(rows):
    cells = [f"{opt:>14}", f"{sd:>14}"] + [f"{d.get(f'pass@{k}',float('nan')):>14.4f}" for k in kvals]
    print("  ".join(cells))

# aggregate by optimizer (mean +/- std)
import statistics as st
agg = defaultdict(lambda: defaultdict(list))
for opt, sd, d in rows:
    for k in kvals:
        if f"pass@{k}" in d: agg[opt][k].append(d[f"pass@{k}"])
print("\n=== Mean +/- std by optimizer (n seeds) ===")
print("  ".join(f"{h:>18}" for h in ["optimizer", "n"] + [f"pass@{k}" for k in kvals]))
for opt in sorted(agg):
    n = len(next(iter(agg[opt].values())))
    cells = [f"{opt:>18}", f"{n:>18}"]
    for k in kvals:
        v = agg[opt][k]
        m = st.mean(v); s = st.stdev(v) if len(v) > 1 else 0.0
        cells.append(f"{m:.4f}+/-{s:.4f}".rjust(18))
    print("  ".join(cells))

if missing:
    print(f"\n[!] Missing {len(missing)} result(s):")
    for m in missing: print("   ", m)
else:
    print("\nAll 12 results present.")
